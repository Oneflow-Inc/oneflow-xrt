/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_COMMON_REGISTER_H_
#define ONEFLOW_XRT_COMMON_REGISTER_H_

#include <string>
#include <unordered_map>

#include "glog/logging.h"

namespace oneflow {
namespace xrt {
namespace common {

template <typename ID, typename K>
class Registry {
 public:
  using Factory = typename ID::Factory;

  static Registry<ID, K>* Global() {
    static Registry<ID, K> registry;
    return &registry;
  }

  bool Has(const K& key) const { return factories_.count(key); }

  bool Register(const K& key, const Factory& value) {
    return factories_.emplace(key, value).second;
  }

  const Factory& Lookup(const K& key) const {
    const auto& it = factories_.find(key);
    if (it == factories_.end()) {
      LOG(FATAL) << "key " << key << " has not been registered";
    }
    return it->second;
  }

 private:
  Registry() = default;
  virtual ~Registry() = default;

 private:
  std::unordered_map<K, Factory> factories_;
};

#define XRT_REGISTER(ID, key, value)                                         \
  common::Registry<ID, std::decay<decltype(key)>::type>::Global()->Register( \
      key, value)

#define XRT_REGISTER_HAS(ID, key) \
  common::Registry<ID, std::decay<decltype(key)>::type>::Global()->Has(key)

#define XRT_REGISTER_LOOKUP(ID, key) \
  common::Registry<ID, std::decay<decltype(key)>::type>::Global()->Lookup(key)

}  // namespace common
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_COMMON_REGISTER_H_
