/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/exec/TopNRowNumber.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

namespace {

std::vector<column_index_t> reorderInputChannels(
    const RowTypePtr& inputType,
    const std::vector<core::FieldAccessTypedExprPtr>& partitionKeys,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys) {
  const auto size = inputType->size();

  std::vector<column_index_t> channels;
  channels.reserve(size);

  std::unordered_set<std::string> keyNames;

  for (const auto& key : partitionKeys) {
    channels.push_back(exprToChannel(key.get(), inputType));
    keyNames.insert(key->name());
  }

  for (const auto& key : sortingKeys) {
    channels.push_back(exprToChannel(key.get(), inputType));
    keyNames.insert(key->name());
  }

  for (auto i = 0; i < size; ++i) {
    if (keyNames.count(inputType->nameOf(i)) == 0) {
      channels.push_back(i);
    }
  }

  return channels;
}

RowTypePtr reorderInputType(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& channels) {
  const auto size = inputType->size();

  VELOX_CHECK_EQ(size, channels.size());

  std::vector<std::string> names;
  names.reserve(size);

  std::vector<TypePtr> types;
  types.reserve(size);

  for (auto channel : channels) {
    names.push_back(inputType->nameOf(channel));
    types.push_back(inputType->childAt(channel));
  }

  return ROW(std::move(names), std::move(types));
}

std::vector<CompareFlags> makeSpillCompareFlags(
    int32_t numPartitionKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  std::vector<CompareFlags> compareFlags;
  compareFlags.reserve(numPartitionKeys + sortingOrders.size());

  for (auto i = 0; i < numPartitionKeys; ++i) {
    compareFlags.push_back({});
  }

  for (const auto& order : sortingOrders) {
    compareFlags.push_back(
        {order.isNullsFirst(), order.isAscending(), false /*equalsOnly*/});
  }

  return compareFlags;
}

// Returns a [start, end) slice of the 'types' vector.
std::vector<TypePtr>
slice(const std::vector<TypePtr>& types, int32_t start, int32_t end) {
  std::vector<TypePtr> result;
  result.reserve(end - start);
  for (auto i = start; i < end; ++i) {
    result.push_back(types[i]);
  }
  return result;
}
} // namespace

TopNRowNumber::TopNRowNumber(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : Operator(
          driverCtx,
          node->outputType(),
          operatorId,
          node->id(),
          "TopNRowNumber",
          node->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId)
              : std::nullopt),
      rankFunction_(node->rankFunction()),
      limit_{node->limit()},
      generateRowNumber_{node->generateRowNumber()},
      numPartitionKeys_{node->partitionKeys().size()},
      numSortingKeys_{node->sortingKeys().size()},
      inputChannels_{reorderInputChannels(
          node->inputType(),
          node->partitionKeys(),
          node->sortingKeys())},
      inputType_{reorderInputType(node->inputType(), inputChannels_)},
      spillCompareFlags_{
          makeSpillCompareFlags(numPartitionKeys_, node->sortingOrders())},
      abandonPartialMinRows_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinRows()),
      abandonPartialMinPct_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinPct()),
      data_(std::make_unique<RowContainer>(
          slice(inputType_->children(), 0, spillCompareFlags_.size()),
          slice(
              inputType_->children(),
              spillCompareFlags_.size(),
              inputType_->size()),
          pool())),
      comparator_(
          inputType_,
          node->sortingKeys(),
          node->sortingOrders(),
          data_.get()),
      decodedVectors_(inputType_->size()) {
  const auto& keys = node->partitionKeys();
  const auto numKeys = keys.size();

  if (numKeys > 0) {
    Accumulator accumulator{
        true,
        sizeof(TopRows),
        false,
        1,
        nullptr,
        [](auto, auto) { VELOX_UNREACHABLE(); },
        [](auto) {}};

    table_ = std::make_unique<HashTable<false>>(
        createVectorHashers(node->inputType(), keys),
        std::vector<Accumulator>{accumulator},
        std::vector<TypePtr>{},
        false, // allowDuplicates
        false, // isJoinBuild
        false, // hasProbedFlag
        0, // minTableSizeForParallelJoinBuild
        pool());
    partitionOffset_ = table_->rows()->columnAt(numKeys).offset();
    lookup_ = std::make_unique<HashLookup>(table_->hashers());
  } else {
    allocator_ = std::make_unique<HashStringAllocator>(pool());
    singlePartition_ = std::make_unique<TopRows>(allocator_.get(), comparator_);
  }

  if (generateRowNumber_) {
    results_.resize(1);
  }
}

void TopNRowNumber::addInput(RowVectorPtr input) {
  if (abandonedPartial_) {
    input_ = std::move(input);
    return;
  }

  const auto numInput = input->size();

  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  if (table_) {
    ensureInputFits(input);

    SelectivityVector rows(numInput);
    table_->prepareForGroupProbe(
        *lookup_, input, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
    table_->groupProbe(*lookup_, BaseHashTable::kNoSpillInputStartPartitionBit);

    // Initialize new partitions.
    initializeNewPartitions();

    // Process input rows. For each row, lookup the partition. If number of rows
    // in that partition is less than limit, add the new row. Otherwise, check
    // if row should replace an existing row or be discarded.
    for (auto i = 0; i < numInput; ++i) {
      auto& partition = partitionAt(lookup_->hits[i]);
      processInputRow(i, partition);
    }

    // It is determined that the TopNRowNumber (as a partial) is not rejecting
    // enough input rows to make the duplicate detection worthwhile. Hence,
    // abandon the processing at this partial TopN and let the final TopN do
    // the processing.
    if (abandonPartialEarly()) {
      abandonedPartial_ = true;
      addRuntimeStat("abandonedPartial", RuntimeCounter(1));

      updateEstimatedOutputRowSize();
      outputBatchSize_ = outputBatchRows(estimatedOutputRowSize_);
      outputRows_.resize(outputBatchSize_);
    }
  } else {
    for (auto i = 0; i < numInput; ++i) {
      processInputRow(i, *singlePartition_);
    }
  }
}

bool TopNRowNumber::abandonPartialEarly() const {
  if (table_ == nullptr || generateRowNumber_ || spiller_ != nullptr) {
    return false;
  }

  const auto numInput = stats_.rlock()->inputPositions;
  if (numInput < abandonPartialMinRows_) {
    return false;
  }

  const auto numOutput = data_->numRows();
  return (100 * numOutput / numInput) >= abandonPartialMinPct_;
}

void TopNRowNumber::initializeNewPartitions() {
  for (auto index : lookup_->newGroups) {
    new (lookup_->hits[index] + partitionOffset_)
        TopRows(table_->stringAllocator(), comparator_);
  }
}

namespace {
template <class T, class S, class C>
S& PriorityQueueVector(std::priority_queue<T, S, C>& q) {
  struct PrivateQueue : private std::priority_queue<T, S, C> {
    static S& Container(std::priority_queue<T, S, C>& q) {
      return q.*&PrivateQueue::c;
    }
  };
  return PrivateQueue::Container(q);
}
} // namespace

bool TopNRowNumber::isDuplicate(
    TopRows& partition,
    const std::vector<DecodedVector>& decodedVectors,
    vector_size_t index) {
  const std::vector<char*, StlAllocator<char*>> partitionRowsVector =
      PriorityQueueVector(partition.rows);
  for (const char* row : partitionRowsVector) {
    if (comparator_.compare(decodedVectors_, index, row) == 0) {
      return true;
    }
  }
  return false;
}

char* TopNRowNumber::removeTopRankRows(TopRows& partition) {
  auto& topRows = partition.rows;
  VELOX_CHECK(!topRows.empty());

  char* topRow = topRows.top();
  topRows.pop();

  while (!topRows.empty()) {
    char* newTopRow = topRows.top();
    if (comparator_.compare(topRow, newTopRow) != 0) {
      return topRow;
    }
    topRows.pop();
  }
  return topRow;
}

vector_size_t TopNRowNumber::numTopRankRows(TopRows& partition) {
  auto& topRows = partition.rows;
  VELOX_CHECK(!topRows.empty());

  std::vector<char*> allTopRows{};
  allTopRows.reserve(topRows.size());

  auto pushTopRows = [&]() -> void {
    for (auto row : allTopRows) {
      topRows.push(row);
    }
  };

  auto popTopRows = [&]() -> void {
    allTopRows.push_back(topRows.top());
    topRows.pop();
  };

  char* topRow = topRows.top();
  popTopRows();
  vector_size_t numRows = 1;
  while (!topRows.empty()) {
    char* newTopRow = topRows.top();
    if (comparator_.compare(topRow, newTopRow) != 0) {
      pushTopRows();
      return numRows;
    }
    numRows += 1;
    popTopRows();
  }

  // All rows in the topRows have the same value. So the top rank = 1.
  pushTopRows();
  return numRows;
}

void TopNRowNumber::processInputRow(vector_size_t index, TopRows& partition) {
  auto& topRows = partition.rows;

  char* newRow = nullptr;
  char* topRow = nullptr;
  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank) {
    if (topRows.empty()) {
      newRow = data_->newRow();
      partition.currentLimit = 1;
    } else {
      topRow = topRows.top();
      if (partition.currentLimit < limit_) {
        newRow = data_->newRow();
        auto result = comparator_.compare(decodedVectors_, index, topRow);
        if (result > 0) {
          partition.currentLimit += 1;
        } else if (result < 0) {
          // This value is greater than the current highest rank. So the new
          // rank is incremented by the number of rows at the top rank.
          partition.currentLimit += numTopRankRows(partition);
        }
      } else {
        auto result = comparator_.compare(decodedVectors_, index, topRow);
        if (result < 0) {
          return;
        }
        if (result == 0) {
          // This row has the same value as the largest value in top.rows.
          // So it needs to be pushed in top.rows. The currentLimit (highest
          // rank) remains unchanged.
          newRow = data_->newRow();
        }
        if (result > 0) {
          topRow = removeTopRankRows(partition);
          newRow = data_->initializeRow(topRow, true /* reuse */);
          // If limit = 1, then the queue becomes empty.
          if (topRows.empty()) {
            partition.currentLimit = 1;
          } else {
            auto numNewTopRankRows = numTopRankRows(partition);
            topRow = topRows.top();
            // Depending on whether the new row is < or = the new top rank row
            // the new top rank changes.
            if (comparator_.compare(decodedVectors_, index, topRow) == 0) {
              partition.currentLimit = topRows.size() - numNewTopRankRows + 1;
            } else {
              partition.currentLimit = topRows.size() - numNewTopRankRows + 2;
            }
          }
        }
      }
    }
  } else {
    if (partition.currentLimit < limit_) {
      newRow = data_->newRow();
      // dense_rank is like row_number with duplicates.
      switch (rankFunction_) {
        case core::TopNRowNumberNode::RankFunction::kDenseRank:
          if (!isDuplicate(partition, decodedVectors_, index)) {
            partition.currentLimit++;
          }
          break;
        case core::TopNRowNumberNode::RankFunction::kRowNumber:
          partition.currentLimit++;
          break;
        default:
          VELOX_UNREACHABLE();
      }
    } else {
      // At this point partition.currentLimit == limit_
      topRow = topRows.top();
      auto result = comparator_.compare(decodedVectors_, index, topRow);
      if (result < 0) {
        // This row will be dropped as its greater than the top ranks.
        return;
      } else if (result == 0) {
        // Same row as last.
        switch (rankFunction_) {
          // This row will have the same rank as the last row. So add it to the
          // priority queue in the accumulator.
          case core::TopNRowNumberNode::RankFunction::kDenseRank:
            newRow = data_->newRow();
            break;
          case core::TopNRowNumberNode::RankFunction::kRowNumber:
            return;
          default:
            VELOX_UNREACHABLE();
        }
      } else {
        // Need to be inserted in queue
        switch (rankFunction_) {
          case core::TopNRowNumberNode::RankFunction::kDenseRank:
            if (!isDuplicate(partition, decodedVectors_, index)) {
              topRow = removeTopRankRows(partition);
            }
            break;
          case core::TopNRowNumberNode::RankFunction::kRowNumber:
            // Replace existing row and reuse its memory.
            topRow = topRows.top();
            topRows.pop();
            break;
          default:
            VELOX_UNREACHABLE();
        }
        newRow = data_->initializeRow(topRow, true /* reuse */);
      }
    }
  }

  for (auto col = 0; col < decodedVectors_.size(); ++col) {
    data_->store(decodedVectors_[col], index, newRow, col);
  }

  topRows.push(newRow);
}

void TopNRowNumber::noMoreInput() {
  Operator::noMoreInput();

  updateEstimatedOutputRowSize();
  outputBatchSize_ = outputBatchRows(estimatedOutputRowSize_);

  if (spiller_ != nullptr) {
    // Spill remaining data to avoid running out of memory while sort-merging
    // spilled data.
    spill();

    VELOX_CHECK_NULL(merge_);
    SpillPartitionSet spillPartitionSet;
    spiller_->finishSpill(spillPartitionSet);
    VELOX_CHECK_EQ(spillPartitionSet.size(), 1);
    merge_ = spillPartitionSet.begin()->second->createOrderedReader(
        spillConfig_->readBufferSize, pool(), &spillStats_);
  } else {
    outputRows_.resize(outputBatchSize_);
  }
}

void TopNRowNumber::updateEstimatedOutputRowSize() {
  const auto optionalRowSize = data_->estimateRowSize();
  if (!optionalRowSize.has_value()) {
    return;
  }

  auto rowSize = optionalRowSize.value();

  if (rowSize && generateRowNumber_) {
    rowSize += sizeof(int64_t);
  }

  if (!estimatedOutputRowSize_.has_value()) {
    estimatedOutputRowSize_ = rowSize;
  } else if (rowSize > estimatedOutputRowSize_.value()) {
    estimatedOutputRowSize_ = rowSize;
  }
}

vector_size_t TopNRowNumber::computeTopRank(TopRows& partition) {
  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank) {
    if (partition.currentLimit > limit_) {
      removeTopRankRows(partition);
      auto numNewTopRankRows = numTopRankRows(partition);
      partition.currentLimit = partition.rows.size() - numNewTopRankRows + 1;
    }
  }

  return partition.currentLimit;
}

TopNRowNumber::TopRows* TopNRowNumber::nextPartition() {
  if (!table_) {
    if (!currentPartitionNumber_) {
      currentPartitionNumber_ = 0;
      nextRank_ = computeTopRank(*singlePartition_);
      numPeers_ = 1;
      return singlePartition_.get();
    }
    return nullptr;
  }

  if (!currentPartitionNumber_) {
    numPartitions_ = table_->listAllRows(
        &partitionIt_,
        partitions_.size(),
        RowContainer::kUnlimited,
        partitions_.data());
    if (numPartitions_ == 0) {
      // No more partitions.
      return nullptr;
    }
    currentPartitionNumber_ = 0;
  } else {
    ++currentPartitionNumber_.value();
    if (currentPartitionNumber_ >= numPartitions_) {
      currentPartitionNumber_.reset();
      return nextPartition();
    }
  }

  auto partition = &partitionAt(partitions_[currentPartitionNumber_.value()]);
  nextRank_ = computeTopRank(*partition);
  numPeers_ = 1;
  return partition;
}

void TopNRowNumber::computeRankInMemory(
    TopRows& partition,
    vector_size_t outputIndex) {
  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    nextRank_ -= 1;
  } else {
    if (comparator_.compare(outputRows_[outputIndex], partition.rows.top()) ==
        0) {
      numPeers_ += 1;
    } else {
      if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kDenseRank) {
        nextRank_ -= 1;
      } else {
        // This is the regular rank function
        nextRank_ -= numPeers_;
      }
      numPeers_ = 1;
    }
  }
}

void TopNRowNumber::appendPartitionRows(
    TopRows& partition,
    vector_size_t numRows,
    vector_size_t outputOffset,
    FlatVector<int64_t>* rankValues) {
  // The partition.rows priority queue pops rows in order of reverse
  // row numbers.
  // auto rank = partition.rows.size();
  for (auto i = 0; i < numRows; ++i) {
    auto index = outputOffset + i;
    if (rankValues) {
      rankValues->set(index, nextRank_);
    }
    outputRows_[index] = partition.rows.top();
    partition.rows.pop();
    if (!partition.rows.empty()) {
      computeRankInMemory(partition, index);
    }
  }
}

RowVectorPtr TopNRowNumber::getOutput() {
  if (finished_) {
    return nullptr;
  }

  if (abandonedPartial_) {
    if (input_ != nullptr) {
      auto output = std::move(input_);
      input_.reset();
      return output;
    }

    // There could be older rows accumulated in 'data_'.
    if (data_->numRows() > 0) {
      return getOutputFromMemory();
    }

    if (noMoreInput_) {
      finished_ = true;
    }
    // There is no data to return at this moment.
    return nullptr;
  }

  if (!noMoreInput_) {
    return nullptr;
  }

  // All the input data is received, so the operator can start producing
  // output.
  RowVectorPtr output;
  if (merge_ != nullptr) {
    output = getOutputFromSpill();
  } else {
    output = getOutputFromMemory();
  }

  if (output == nullptr) {
    finished_ = true;
  }

  return output;
}

RowVectorPtr TopNRowNumber::getOutputFromMemory() {
  VELOX_CHECK_GT(outputBatchSize_, 0);

  // Loop over partitions and emit sorted rows along with row numbers.
  auto output =
      BaseVector::create<RowVector>(outputType_, outputBatchSize_, pool());
  FlatVector<int64_t>* rowNumbers = nullptr;
  if (generateRowNumber_) {
    rowNumbers = output->children().back()->as<FlatVector<int64_t>>();
  }

  vector_size_t offset = 0;
  // Continue to output as many remaining partitions as possible.
  while (offset < outputBatchSize_) {
    // No previous partition to output (since this is the first partition).
    if (!currentPartition_) {
      currentPartition_ = nextPartition();
      if (!currentPartition_) {
        break;
      }
    }

    auto numOutputRowsLeft = outputBatchSize_ - offset;
    if (currentPartition_->rows.size() > numOutputRowsLeft) {
      // Only a partial partition can be output in this getOutput() call.
      // Output as many rows as possible.
      appendPartitionRows(
          *currentPartition_, numOutputRowsLeft, offset, rowNumbers);
      offset += numOutputRowsLeft;
      break;
    }

    // Add all partition rows.
    auto numPartitionRows = currentPartition_->rows.size();
    appendPartitionRows(
        *currentPartition_, numPartitionRows, offset, rowNumbers);
    offset += numPartitionRows;

    // Move to the next partition.
    currentPartition_ = nextPartition();
  }

  if (offset == 0) {
    data_->clear();
    if (table_ != nullptr) {
      table_->clear(true);
    }
    pool()->release();
    return nullptr;
  }

  if (rowNumbers) {
    rowNumbers->resize(offset);
  }
  output->resize(offset);

  for (int i = 0; i < inputChannels_.size(); ++i) {
    data_->extractColumn(
        outputRows_.data(), offset, i, output->childAt(inputChannels_[i]));
  }

  return output;
}

bool TopNRowNumber::isNewPartition(
    const RowVectorPtr& output,
    vector_size_t index,
    SpillMergeStream* next) {
  VELOX_CHECK_GT(index, 0);

  for (auto i = 0; i < numPartitionKeys_; ++i) {
    if (!output->childAt(inputChannels_[i])
             ->equalValueAt(
                 next->current().childAt(i).get(),
                 index - 1,
                 next->currentIndex())) {
      return true;
    }
  }
  return false;
}

bool TopNRowNumber::isNewPeer(
    const RowVectorPtr& output,
    vector_size_t index,
    SpillMergeStream* next) {
  VELOX_CHECK_GT(index, 0);

  for (auto i = numPartitionKeys_; i < numPartitionKeys_ + numSortingKeys_;
       ++i) {
    if (!output->childAt(inputChannels_[i])
             ->equalValueAt(
                 next->current().childAt(i).get(),
                 index - 1,
                 next->currentIndex())) {
      return true;
    }
  }
  return false;
}

void TopNRowNumber::setupNextOutput(
    const RowVectorPtr& output,
    int32_t currentRank,
    int32_t numPeers) {
  auto* lookAhead = merge_->next();
  if (lookAhead == nullptr) {
    nextRank_ = 1;
    numPeers_ = 1;
    return;
  }

  if (isNewPartition(output, output->size(), lookAhead)) {
    nextRank_ = 1;
    numPeers_ = 1;
    return;
  }

  nextRank_ = currentRank;
  numPeers_ = numPeers;
  // This row belongs to the same partition as the previous row. However,
  // it should be determined if it is a peer row as well. If peer, then rank
  // is not increased.
  if (isNewPeer(output, output->size(), lookAhead)) {
    if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kDenseRank) {
      nextRank_ += 1;
      numPeers_ = 1;
    } else if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank) {
      nextRank_ += numPeers_;
      numPeers_ = 1;
    }
  } else {
    numPeers_ += 1;
  }

  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    nextRank_ += 1;
  }

  if (nextRank_ <= limit_) {
    return;
  }

  // Skip remaining rows for this partition.
  lookAhead->pop();

  while (auto* next = merge_->next()) {
    if (isNewPartition(output, output->size(), next)) {
      nextRank_ = 1;
      numPeers_ = 1;
      return;
    }
    next->pop();
  }

  // This partition is the last partition.
  nextRank_ = 1;
  numPeers_ = 1;
}

RowVectorPtr TopNRowNumber::getOutputFromSpill() {
  VELOX_CHECK_NOT_NULL(merge_);

  // merge_->next() produces data sorted by partition keys, then sorting keys.
  // All rows from the same partition will appear together.
  // We'll identify partition boundaries by comparing partition keys of the
  // current row with the previous row. When new partition starts, we'll reset
  // row number to zero. Once row number reaches the 'limit_', we'll start
  // dropping rows until the next partition starts.
  // We'll emit output every time we accumulate 'outputBatchSize_' rows.
  auto output =
      BaseVector::create<RowVector>(outputType_, outputBatchSize_, pool());
  FlatVector<int64_t>* rankValues = nullptr;
  if (generateRowNumber_) {
    rankValues = output->children().back()->as<FlatVector<int64_t>>();
  }

  // In the case of rank and dense_rank, output values are the same for all
  // peer rows. So track the last row of each output block for detecting
  // peer row changes across output blocks.
  // auto lastRow = BaseVector::create<RowVector>(outputType_, 1, pool());

  // Index of the next row to append to output.
  vector_size_t index = 0;

  // Row number of the next row in the current partition.
  vector_size_t rank = nextRank_;
  VELOX_CHECK_LE(rank, limit_);
  // Tracks the number of peers of the current row seen thus far.
  // This is used to increment ranks for the rank function.
  vector_size_t numPeers = numPeers_;
  for (;;) {
    auto next = merge_->next();
    if (next == nullptr) {
      break;
    }

    if (index > 0) {
      // Check if this row comes from a new partition.
      if (isNewPartition(output, index, next)) {
        rank = 1;
        numPeers = 1;
      } else {
        // This row is the same partition as the previous. Check if it is a
        // peer or not. If it is a peer then the rank values change.
        if (isNewPeer(output, index, next)) {
          if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank) {
            rank += numPeers;
          } else if (
              rankFunction_ ==
              core::TopNRowNumberNode::RankFunction::kDenseRank) {
            rank += 1;
          }
          numPeers = 1;
        } else {
          numPeers += 1;
        }

        if (rankFunction_ ==
            core::TopNRowNumberNode::RankFunction::kRowNumber) {
          rank += 1;
        }
      }
    }

    // Copy this row to the output buffer if this partition has
    // < limit_ rows output.
    if (rank <= limit_) {
      for (auto i = 0; i < inputChannels_.size(); ++i) {
        output->childAt(inputChannels_[i])
            ->copy(
                next->current().childAt(i).get(),
                index,
                next->currentIndex(),
                1);
      }
      if (rankValues) {
        rankValues->set(index, rank);
      }
      ++index;
    }

    // Pop this row from the spill.
    next->pop();

    if (index == outputBatchSize_) {
      // This is the last row for this output batch.
      // Prepare the next batch :
      // i) If 'limit_' is reached for this partition, then skip the rows
      // until the next partition.
      // ii) If the next row is from a new partition, then reset rank_.
      setupNextOutput(output, rank, numPeers);

      return output;
    }
  }

  // At this point, all rows are read from the spill merge stream.
  // (Note : The previous loop returns directly when the output buffer
  // is filled).
  if (index > 0) {
    output->resize(index);
  } else {
    output = nullptr;
  }

  finished_ = true;
  return output;
}

bool TopNRowNumber::isFinished() {
  return finished_;
}

void TopNRowNumber::close() {
  Operator::close();

  SCOPE_EXIT {
    table_.reset();
    singlePartition_.reset();
    data_.reset();
    allocator_.reset();
  };

  if (table_ == nullptr) {
    return;
  }

  partitionIt_.reset();
  partitions_.resize(1'000);
  while (auto numPartitions = table_->listAllRows(
             &partitionIt_,
             partitions_.size(),
             RowContainer::kUnlimited,
             partitions_.data())) {
    for (auto i = 0; i < numPartitions; ++i) {
      std::destroy_at(
          reinterpret_cast<TopRows*>(partitions_[i] + partitionOffset_));
    }
  }
}

void TopNRowNumber::reclaim(
    uint64_t /*targetBytes*/,
    memory::MemoryReclaimer::Stats& stats) {
  VELOX_CHECK(canReclaim());
  VELOX_CHECK(!nonReclaimableSection_);

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  if (noMoreInput_) {
    ++stats.numNonReclaimableAttempts;
    // TODO Add support for spilling after noMoreInput().
    LOG(WARNING)
        << "Can't reclaim from topNRowNumber operator which has started producing output: "
        << pool()->name() << ", usage: " << succinctBytes(pool()->usedBytes())
        << ", reservation: " << succinctBytes(pool()->reservedBytes());
    return;
  }

  if (abandonedPartial_) {
    return;
  }

  spill();
}

void TopNRowNumber::ensureInputFits(const RowVectorPtr& input) {
  if (!spillEnabled()) {
    // Spilling is disabled.
    return;
  }

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool()->name())) {
    spill();
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto outOfLineBytesPerRow = outOfLineBytes / data_->numRows();

  const auto currentUsage = pool()->usedBytes();
  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool()->availableReservation();
  const auto tableIncrementBytes = table_->hashTableSizeIncrease(input->size());
  const auto incrementBytes =
      data_->sizeIncrement(
          input->size(), outOfLineBytesPerRow * input->size()) +
      tableIncrementBytes;

  // First to check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    if ((tableIncrementBytes == 0) && (freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      // Enough free rows for input rows and enough variable length free space.
      return;
    }
  }

  // Check if we can increase reservation. The increment is the largest of twice
  // the maximum increment from this input and 'spillableReservationGrowthPct_'
  // of the current memory usage.
  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }

  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", usage: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes());
}

void TopNRowNumber::spill() {
  if (spiller_ == nullptr) {
    setupSpiller();
  }

  updateEstimatedOutputRowSize();

  spiller_->spill();
  table_->clear(true);
  data_->clear();
  pool()->release();
}

void TopNRowNumber::setupSpiller() {
  VELOX_CHECK_NULL(spiller_);
  VELOX_CHECK(spillConfig_.has_value());

  spiller_ = std::make_unique<Spiller>(
      // TODO Replace Spiller::Type::kOrderBy.
      Spiller::Type::kOrderByInput,
      data_.get(),
      inputType_,
      spillCompareFlags_.size(),
      spillCompareFlags_,
      &spillConfig_.value(),
      &spillStats_);
}
} // namespace facebook::velox::exec
