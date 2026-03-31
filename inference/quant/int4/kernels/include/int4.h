#pragma once
#include <cutlass/subbyte_reference.h>

namespace cutlass {

template <typename Element_, typename Storage_>
class MySubbyteReference {
 public:
  using Element = Element_;
  using Storage = Storage_;
  using StoragePointer = Storage *;

  static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value,
                "Size of Element must not be greater than Storage.");
  static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value),
                "Storage must be divisible by Element");

  constexpr static int const kElementsPerVector =
      sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

 private:
  Storage const kMask =
      ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value)
           ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1)
           : ~Storage(0));

  StoragePointer ptr_;
  int offset_;

 public:
  CUTLASS_HOST_DEVICE
  MySubbyteReference() : ptr_(nullptr), offset_(0) {}

  CUTLASS_HOST_DEVICE
  MySubbyteReference(Element *ptr, int64_t offset)
      : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
    int64_t offset_in_vectors = offset / kElementsPerVector;
    int64_t offset_in_elements = offset % kElementsPerVector;
    ptr_ += offset_in_vectors;
    offset_ = int(offset_in_elements);
  }

  CUTLASS_HOST_DEVICE
  MySubbyteReference(Element *ptr = nullptr) : MySubbyteReference(ptr, 0) {}

  CUTLASS_HOST_DEVICE
  StoragePointer storage_pointer() const { return ptr_; }

  CUTLASS_HOST_DEVICE
  Element *operator&() const { return reinterpret_cast<Element *>(ptr_); }

  CUTLASS_HOST_DEVICE
  int element_offset() const { return offset_; }

  CUTLASS_HOST_DEVICE
  Element get() const {
    Storage item =
        Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
    return reinterpret_cast<Element const &>(item);
  }

  CUTLASS_HOST_DEVICE
  MySubbyteReference &set(Element const &x) {
    Storage item = (reinterpret_cast<Storage const &>(x) & kMask);
    Storage kUpdateMask =
        Storage(~(kMask << (offset_ * cutlass::sizeof_bits<Element>::value)));
    Storage new_bits =
        Storage(item << (offset_ * cutlass::sizeof_bits<Element>::value));

    Storage original = (*ptr_);
    Storage updated = Storage((original & kUpdateMask) | new_bits);
    *ptr_ = updated;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  operator Element() const { return get(); }

  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(Element const &x) { return set(x); }

  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(MySubbyteReference const &x) {
    return set(x.get());
  }

  CUTLASS_HOST_DEVICE
  MySubbyteReference &operator=(
      ConstSubbyteReference<Element, Storage> const &x) {
    return set(x.get());
  }
};

}  // namespace cutlass

using Int4Subbyte = cutlass::MySubbyteReference<cutlass::int4b_t, uint8_t>;
using Int4Storage = Int4Subbyte::Storage;
constexpr const uint32_t kElementsPerVector =
    cutlass::sizeof_bits<Int4Storage>::value /
    cutlass::sizeof_bits<cutlass::int4b_t>::value;

