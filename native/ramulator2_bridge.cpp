#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#ifndef GALA_RAMULATOR_VERSION
#define GALA_RAMULATOR_VERSION "unknown"
#endif

namespace {

thread_local std::string last_error;

class Bridge {
 public:
  explicit Bridge(const char* configuration) {
    if (configuration == nullptr || *configuration == '\0') {
      throw std::invalid_argument("Ramulator configuration path is empty");
    }
    const Ramulator::ConfigNode document =
        Ramulator::Config::parse_config_file(configuration);
    frontend_.reset(Ramulator::Factory::create_frontend(document));
    memory_.reset(Ramulator::Factory::create_memory_system(document));
    frontend_->connect_memory_system(memory_.get());
    memory_->connect_frontend(frontend_.get());
    if (frontend_->get_clock_ratio() != 1 || memory_->get_clock_ratio() != 1) {
      throw std::runtime_error("Ramulator frontend and memory clock ratios must be one");
    }
    transaction_bytes_ = memory_->get_tx_bytes();
    if (transaction_bytes_ <= 0) {
      throw std::runtime_error("Ramulator returned an invalid transaction size");
    }
  }

  bool try_issue(std::uint64_t address, bool write, std::uint64_t request_id) {
    const int type = write ? Ramulator::Request::Type::Write
                           : Ramulator::Request::Type::Read;
    return frontend_->receive_external_requests(
        type, static_cast<Ramulator::Addr_t>(address), 0,
        [this, request_id](Ramulator::Request&) {
          completions_.push_back(request_id);
        },
        transaction_bytes_);
  }

  void enqueue_group(std::uint64_t group_id, const std::uint64_t* addresses,
                     const std::uint64_t* request_ids, std::size_t count,
                     bool write) {
    if (addresses == nullptr || request_ids == nullptr || count == 0) {
      throw std::invalid_argument("Ramulator request group is empty");
    }
    if (pending_groups_.contains(group_id)) {
      throw std::invalid_argument("Ramulator request group is duplicated");
    }
    PendingGroup group;
    group.write = write;
    for (std::size_t index = 0; index < count; ++index) {
      group.beats.push_back({addresses[index], request_ids[index]});
    }
    pending_groups_.emplace(group_id, std::move(group));
    issue_waiting();
  }

  void tick() {
    issue_waiting();
    memory_->tick();
  }

  std::size_t completion_count() const { return completions_.size(); }

  std::size_t drain(std::uint64_t* output, std::size_t capacity) {
    if (capacity < completions_.size()) {
      throw std::invalid_argument("completion output buffer is too small");
    }
    const std::size_t count = completions_.size();
    for (std::size_t index = 0; index < count; ++index) {
      output[index] = completions_.front();
      completions_.pop_front();
    }
    return count;
  }

  int transaction_bytes() const { return transaction_bytes_; }

 private:
  struct PendingBeat {
    std::uint64_t address;
    std::uint64_t request_id;
  };

  struct PendingGroup {
    bool write{};
    std::deque<PendingBeat> beats;
  };

  void issue_waiting() {
    auto group = pending_groups_.begin();
    while (group != pending_groups_.end()) {
      while (!group->second.beats.empty()) {
        const PendingBeat& beat = group->second.beats.front();
        if (!try_issue(beat.address, group->second.write, beat.request_id)) {
          break;
        }
        group->second.beats.pop_front();
      }
      if (group->second.beats.empty()) {
        group = pending_groups_.erase(group);
      } else {
        ++group;
      }
    }
  }

  std::unique_ptr<Ramulator::IFrontEnd> frontend_;
  std::unique_ptr<Ramulator::IMemorySystem> memory_;
  std::map<std::uint64_t, PendingGroup> pending_groups_;
  std::deque<std::uint64_t> completions_;
  int transaction_bytes_{};
};

template <typename Function, typename Result>
Result guard(Function&& function, Result failure) noexcept {
  try {
    last_error.clear();
    return function();
  } catch (const std::exception& error) {
    last_error = error.what();
    return failure;
  } catch (...) {
    last_error = "unknown native Ramulator bridge failure";
    return failure;
  }
}

}  // namespace

extern "C" {

void* gala_ramulator_create(const char* configuration) noexcept {
  return guard([&]() -> void* { return new Bridge(configuration); },
               static_cast<void*>(nullptr));
}

void gala_ramulator_destroy(void* handle) noexcept {
  delete static_cast<Bridge*>(handle);
}

int gala_ramulator_try_issue(void* handle, std::uint64_t address, int write,
                             std::uint64_t request_id) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        return static_cast<Bridge*>(handle)->try_issue(address, write != 0,
                                                       request_id)
                   ? 1
                   : 0;
      },
      -1);
}

int gala_ramulator_enqueue_group(void* handle, std::uint64_t group_id,
                                 const std::uint64_t* addresses,
                                 const std::uint64_t* request_ids,
                                 std::size_t count, int write) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        static_cast<Bridge*>(handle)->enqueue_group(
            group_id, addresses, request_ids, count, write != 0);
        return 0;
      },
      -1);
}

int gala_ramulator_tick(void* handle) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        static_cast<Bridge*>(handle)->tick();
        return 0;
      },
      -1);
}

std::size_t gala_ramulator_completion_count(void* handle) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        return static_cast<Bridge*>(handle)->completion_count();
      },
      static_cast<std::size_t>(-1));
}

std::size_t gala_ramulator_drain(void* handle, std::uint64_t* output,
                                 std::size_t capacity) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        return static_cast<Bridge*>(handle)->drain(output, capacity);
      },
      static_cast<std::size_t>(-1));
}

int gala_ramulator_transaction_bytes(void* handle) noexcept {
  return guard(
      [&]() {
        if (handle == nullptr) throw std::invalid_argument("bridge handle is null");
        return static_cast<Bridge*>(handle)->transaction_bytes();
      },
      -1);
}

const char* gala_ramulator_version() noexcept { return GALA_RAMULATOR_VERSION; }

const char* gala_ramulator_last_error() noexcept { return last_error.c_str(); }

}  // extern "C"
