// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: HypothesisList.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "HypothesisList.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace {

const ::google::protobuf::Descriptor* HypothesisList_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  HypothesisList_reflection_ = NULL;
const ::google::protobuf::Descriptor* HypothesisList_ObjectHypothesis_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  HypothesisList_ObjectHypothesis_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_HypothesisList_2eproto() GOOGLE_ATTRIBUTE_COLD;
void protobuf_AssignDesc_HypothesisList_2eproto() {
  protobuf_AddDesc_HypothesisList_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "HypothesisList.proto");
  GOOGLE_CHECK(file != NULL);
  HypothesisList_descriptor_ = file->message_type(0);
  static const int HypothesisList_offsets_[1] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList, hyp_),
  };
  HypothesisList_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      HypothesisList_descriptor_,
      HypothesisList::internal_default_instance(),
      HypothesisList_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList, _has_bits_),
      -1,
      -1,
      sizeof(HypothesisList),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList, _internal_metadata_));
  HypothesisList_ObjectHypothesis_descriptor_ = HypothesisList_descriptor_->nested_type(0);
  static const int HypothesisList_ObjectHypothesis_offsets_[5] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, x_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, y_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, scale_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, score_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, flip_),
  };
  HypothesisList_ObjectHypothesis_reflection_ =
    ::google::protobuf::internal::GeneratedMessageReflection::NewGeneratedMessageReflection(
      HypothesisList_ObjectHypothesis_descriptor_,
      HypothesisList_ObjectHypothesis::internal_default_instance(),
      HypothesisList_ObjectHypothesis_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, _has_bits_),
      -1,
      -1,
      sizeof(HypothesisList_ObjectHypothesis),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(HypothesisList_ObjectHypothesis, _internal_metadata_));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_HypothesisList_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      HypothesisList_descriptor_, HypothesisList::internal_default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
      HypothesisList_ObjectHypothesis_descriptor_, HypothesisList_ObjectHypothesis::internal_default_instance());
}

}  // namespace

void protobuf_ShutdownFile_HypothesisList_2eproto() {
  HypothesisList_default_instance_.Shutdown();
  delete HypothesisList_reflection_;
  HypothesisList_ObjectHypothesis_default_instance_.Shutdown();
  delete HypothesisList_ObjectHypothesis_reflection_;
}

void protobuf_InitDefaults_HypothesisList_2eproto_impl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  HypothesisList_default_instance_.DefaultConstruct();
  HypothesisList_ObjectHypothesis_default_instance_.DefaultConstruct();
  HypothesisList_default_instance_.get_mutable()->InitAsDefaultInstance();
  HypothesisList_ObjectHypothesis_default_instance_.get_mutable()->InitAsDefaultInstance();
}

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_InitDefaults_HypothesisList_2eproto_once_);
void protobuf_InitDefaults_HypothesisList_2eproto() {
  ::google::protobuf::GoogleOnceInit(&protobuf_InitDefaults_HypothesisList_2eproto_once_,
                 &protobuf_InitDefaults_HypothesisList_2eproto_impl);
}
void protobuf_AddDesc_HypothesisList_2eproto_impl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  protobuf_InitDefaults_HypothesisList_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\024HypothesisList.proto\"\234\001\n\016HypothesisLis"
    "t\022-\n\003hyp\030\001 \003(\0132 .HypothesisList.ObjectHy"
    "pothesis\032[\n\020ObjectHypothesis\022\t\n\001x\030\001 \001(\002\022"
    "\t\n\001y\030\002 \001(\002\022\r\n\005scale\030\003 \001(\002\022\r\n\005score\030\004 \001(\002"
    "\022\023\n\004flip\030\005 \001(\010:\005false", 181);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "HypothesisList.proto", &protobuf_RegisterTypes);
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_HypothesisList_2eproto);
}

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AddDesc_HypothesisList_2eproto_once_);
void protobuf_AddDesc_HypothesisList_2eproto() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AddDesc_HypothesisList_2eproto_once_,
                 &protobuf_AddDesc_HypothesisList_2eproto_impl);
}
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_HypothesisList_2eproto {
  StaticDescriptorInitializer_HypothesisList_2eproto() {
    protobuf_AddDesc_HypothesisList_2eproto();
  }
} static_descriptor_initializer_HypothesisList_2eproto_;

namespace {

static void MergeFromFail(int line) GOOGLE_ATTRIBUTE_COLD GOOGLE_ATTRIBUTE_NORETURN;
static void MergeFromFail(int line) {
  ::google::protobuf::internal::MergeFromFail(__FILE__, line);
}

}  // namespace


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int HypothesisList_ObjectHypothesis::kXFieldNumber;
const int HypothesisList_ObjectHypothesis::kYFieldNumber;
const int HypothesisList_ObjectHypothesis::kScaleFieldNumber;
const int HypothesisList_ObjectHypothesis::kScoreFieldNumber;
const int HypothesisList_ObjectHypothesis::kFlipFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

HypothesisList_ObjectHypothesis::HypothesisList_ObjectHypothesis()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (this != internal_default_instance()) protobuf_InitDefaults_HypothesisList_2eproto();
  SharedCtor();
  // @@protoc_insertion_point(constructor:HypothesisList.ObjectHypothesis)
}

void HypothesisList_ObjectHypothesis::InitAsDefaultInstance() {
}

HypothesisList_ObjectHypothesis::HypothesisList_ObjectHypothesis(const HypothesisList_ObjectHypothesis& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  UnsafeMergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:HypothesisList.ObjectHypothesis)
}

void HypothesisList_ObjectHypothesis::SharedCtor() {
  _cached_size_ = 0;
  ::memset(&x_, 0, reinterpret_cast<char*>(&flip_) -
    reinterpret_cast<char*>(&x_) + sizeof(flip_));
}

HypothesisList_ObjectHypothesis::~HypothesisList_ObjectHypothesis() {
  // @@protoc_insertion_point(destructor:HypothesisList.ObjectHypothesis)
  SharedDtor();
}

void HypothesisList_ObjectHypothesis::SharedDtor() {
}

void HypothesisList_ObjectHypothesis::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* HypothesisList_ObjectHypothesis::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return HypothesisList_ObjectHypothesis_descriptor_;
}

const HypothesisList_ObjectHypothesis& HypothesisList_ObjectHypothesis::default_instance() {
  protobuf_InitDefaults_HypothesisList_2eproto();
  return *internal_default_instance();
}

::google::protobuf::internal::ExplicitlyConstructed<HypothesisList_ObjectHypothesis> HypothesisList_ObjectHypothesis_default_instance_;

HypothesisList_ObjectHypothesis* HypothesisList_ObjectHypothesis::New(::google::protobuf::Arena* arena) const {
  HypothesisList_ObjectHypothesis* n = new HypothesisList_ObjectHypothesis;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void HypothesisList_ObjectHypothesis::Clear() {
// @@protoc_insertion_point(message_clear_start:HypothesisList.ObjectHypothesis)
#if defined(__clang__)
#define ZR_HELPER_(f) \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Winvalid-offsetof\"") \
  __builtin_offsetof(HypothesisList_ObjectHypothesis, f) \
  _Pragma("clang diagnostic pop")
#else
#define ZR_HELPER_(f) reinterpret_cast<char*>(\
  &reinterpret_cast<HypothesisList_ObjectHypothesis*>(16)->f)
#endif

#define ZR_(first, last) do {\
  ::memset(&(first), 0,\
           ZR_HELPER_(last) - ZR_HELPER_(first) + sizeof(last));\
} while (0)

  ZR_(x_, flip_);

#undef ZR_HELPER_
#undef ZR_

  _has_bits_.Clear();
  if (_internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->Clear();
  }
}

bool HypothesisList_ObjectHypothesis::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:HypothesisList.ObjectHypothesis)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional float x = 1;
      case 1: {
        if (tag == 13) {
          set_has_x();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(21)) goto parse_y;
        break;
      }

      // optional float y = 2;
      case 2: {
        if (tag == 21) {
         parse_y:
          set_has_y();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(29)) goto parse_scale;
        break;
      }

      // optional float scale = 3;
      case 3: {
        if (tag == 29) {
         parse_scale:
          set_has_scale();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &scale_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(37)) goto parse_score;
        break;
      }

      // optional float score = 4;
      case 4: {
        if (tag == 37) {
         parse_score:
          set_has_score();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &score_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(40)) goto parse_flip;
        break;
      }

      // optional bool flip = 5 [default = false];
      case 5: {
        if (tag == 40) {
         parse_flip:
          set_has_flip();
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &flip_)));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:HypothesisList.ObjectHypothesis)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:HypothesisList.ObjectHypothesis)
  return false;
#undef DO_
}

void HypothesisList_ObjectHypothesis::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:HypothesisList.ObjectHypothesis)
  // optional float x = 1;
  if (has_x()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(1, this->x(), output);
  }

  // optional float y = 2;
  if (has_y()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->y(), output);
  }

  // optional float scale = 3;
  if (has_scale()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(3, this->scale(), output);
  }

  // optional float score = 4;
  if (has_score()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(4, this->score(), output);
  }

  // optional bool flip = 5 [default = false];
  if (has_flip()) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(5, this->flip(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:HypothesisList.ObjectHypothesis)
}

::google::protobuf::uint8* HypothesisList_ObjectHypothesis::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:HypothesisList.ObjectHypothesis)
  // optional float x = 1;
  if (has_x()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(1, this->x(), target);
  }

  // optional float y = 2;
  if (has_y()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->y(), target);
  }

  // optional float scale = 3;
  if (has_scale()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(3, this->scale(), target);
  }

  // optional float score = 4;
  if (has_score()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(4, this->score(), target);
  }

  // optional bool flip = 5 [default = false];
  if (has_flip()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(5, this->flip(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:HypothesisList.ObjectHypothesis)
  return target;
}

size_t HypothesisList_ObjectHypothesis::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:HypothesisList.ObjectHypothesis)
  size_t total_size = 0;

  if (_has_bits_[0 / 32] & 31u) {
    // optional float x = 1;
    if (has_x()) {
      total_size += 1 + 4;
    }

    // optional float y = 2;
    if (has_y()) {
      total_size += 1 + 4;
    }

    // optional float scale = 3;
    if (has_scale()) {
      total_size += 1 + 4;
    }

    // optional float score = 4;
    if (has_score()) {
      total_size += 1 + 4;
    }

    // optional bool flip = 5 [default = false];
    if (has_flip()) {
      total_size += 1 + 1;
    }

  }
  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void HypothesisList_ObjectHypothesis::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:HypothesisList.ObjectHypothesis)
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const HypothesisList_ObjectHypothesis* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const HypothesisList_ObjectHypothesis>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:HypothesisList.ObjectHypothesis)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:HypothesisList.ObjectHypothesis)
    UnsafeMergeFrom(*source);
  }
}

void HypothesisList_ObjectHypothesis::MergeFrom(const HypothesisList_ObjectHypothesis& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:HypothesisList.ObjectHypothesis)
  if (GOOGLE_PREDICT_TRUE(&from != this)) {
    UnsafeMergeFrom(from);
  } else {
    MergeFromFail(__LINE__);
  }
}

void HypothesisList_ObjectHypothesis::UnsafeMergeFrom(const HypothesisList_ObjectHypothesis& from) {
  GOOGLE_DCHECK(&from != this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_x()) {
      set_x(from.x());
    }
    if (from.has_y()) {
      set_y(from.y());
    }
    if (from.has_scale()) {
      set_scale(from.scale());
    }
    if (from.has_score()) {
      set_score(from.score());
    }
    if (from.has_flip()) {
      set_flip(from.flip());
    }
  }
  if (from._internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::UnknownFieldSet::MergeToInternalMetdata(
      from.unknown_fields(), &_internal_metadata_);
  }
}

void HypothesisList_ObjectHypothesis::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:HypothesisList.ObjectHypothesis)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void HypothesisList_ObjectHypothesis::CopyFrom(const HypothesisList_ObjectHypothesis& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:HypothesisList.ObjectHypothesis)
  if (&from == this) return;
  Clear();
  UnsafeMergeFrom(from);
}

bool HypothesisList_ObjectHypothesis::IsInitialized() const {

  return true;
}

void HypothesisList_ObjectHypothesis::Swap(HypothesisList_ObjectHypothesis* other) {
  if (other == this) return;
  InternalSwap(other);
}
void HypothesisList_ObjectHypothesis::InternalSwap(HypothesisList_ObjectHypothesis* other) {
  std::swap(x_, other->x_);
  std::swap(y_, other->y_);
  std::swap(scale_, other->scale_);
  std::swap(score_, other->score_);
  std::swap(flip_, other->flip_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata HypothesisList_ObjectHypothesis::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = HypothesisList_ObjectHypothesis_descriptor_;
  metadata.reflection = HypothesisList_ObjectHypothesis_reflection_;
  return metadata;
}


// -------------------------------------------------------------------

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int HypothesisList::kHypFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

HypothesisList::HypothesisList()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (this != internal_default_instance()) protobuf_InitDefaults_HypothesisList_2eproto();
  SharedCtor();
  // @@protoc_insertion_point(constructor:HypothesisList)
}

void HypothesisList::InitAsDefaultInstance() {
}

HypothesisList::HypothesisList(const HypothesisList& from)
  : ::google::protobuf::Message(),
    _internal_metadata_(NULL) {
  SharedCtor();
  UnsafeMergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:HypothesisList)
}

void HypothesisList::SharedCtor() {
  _cached_size_ = 0;
}

HypothesisList::~HypothesisList() {
  // @@protoc_insertion_point(destructor:HypothesisList)
  SharedDtor();
}

void HypothesisList::SharedDtor() {
}

void HypothesisList::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* HypothesisList::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return HypothesisList_descriptor_;
}

const HypothesisList& HypothesisList::default_instance() {
  protobuf_InitDefaults_HypothesisList_2eproto();
  return *internal_default_instance();
}

::google::protobuf::internal::ExplicitlyConstructed<HypothesisList> HypothesisList_default_instance_;

HypothesisList* HypothesisList::New(::google::protobuf::Arena* arena) const {
  HypothesisList* n = new HypothesisList;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void HypothesisList::Clear() {
// @@protoc_insertion_point(message_clear_start:HypothesisList)
  hyp_.Clear();
  _has_bits_.Clear();
  if (_internal_metadata_.have_unknown_fields()) {
    mutable_unknown_fields()->Clear();
  }
}

bool HypothesisList::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:HypothesisList)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .HypothesisList.ObjectHypothesis hyp = 1;
      case 1: {
        if (tag == 10) {
          DO_(input->IncrementRecursionDepth());
         parse_loop_hyp:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtualNoRecursionDepth(
                input, add_hyp()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(10)) goto parse_loop_hyp;
        input->UnsafeDecrementRecursionDepth();
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:HypothesisList)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:HypothesisList)
  return false;
#undef DO_
}

void HypothesisList::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:HypothesisList)
  // repeated .HypothesisList.ObjectHypothesis hyp = 1;
  for (unsigned int i = 0, n = this->hyp_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->hyp(i), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:HypothesisList)
}

::google::protobuf::uint8* HypothesisList::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:HypothesisList)
  // repeated .HypothesisList.ObjectHypothesis hyp = 1;
  for (unsigned int i = 0, n = this->hyp_size(); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageNoVirtualToArray(
        1, this->hyp(i), false, target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:HypothesisList)
  return target;
}

size_t HypothesisList::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:HypothesisList)
  size_t total_size = 0;

  // repeated .HypothesisList.ObjectHypothesis hyp = 1;
  {
    unsigned int count = this->hyp_size();
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->hyp(i));
    }
  }

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void HypothesisList::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:HypothesisList)
  if (GOOGLE_PREDICT_FALSE(&from == this)) MergeFromFail(__LINE__);
  const HypothesisList* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const HypothesisList>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:HypothesisList)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:HypothesisList)
    UnsafeMergeFrom(*source);
  }
}

void HypothesisList::MergeFrom(const HypothesisList& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:HypothesisList)
  if (GOOGLE_PREDICT_TRUE(&from != this)) {
    UnsafeMergeFrom(from);
  } else {
    MergeFromFail(__LINE__);
  }
}

void HypothesisList::UnsafeMergeFrom(const HypothesisList& from) {
  GOOGLE_DCHECK(&from != this);
  hyp_.MergeFrom(from.hyp_);
  if (from._internal_metadata_.have_unknown_fields()) {
    ::google::protobuf::UnknownFieldSet::MergeToInternalMetdata(
      from.unknown_fields(), &_internal_metadata_);
  }
}

void HypothesisList::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:HypothesisList)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void HypothesisList::CopyFrom(const HypothesisList& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:HypothesisList)
  if (&from == this) return;
  Clear();
  UnsafeMergeFrom(from);
}

bool HypothesisList::IsInitialized() const {

  return true;
}

void HypothesisList::Swap(HypothesisList* other) {
  if (other == this) return;
  InternalSwap(other);
}
void HypothesisList::InternalSwap(HypothesisList* other) {
  hyp_.UnsafeArenaSwap(&other->hyp_);
  std::swap(_has_bits_[0], other->_has_bits_[0]);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata HypothesisList::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = HypothesisList_descriptor_;
  metadata.reflection = HypothesisList_reflection_;
  return metadata;
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// HypothesisList_ObjectHypothesis

// optional float x = 1;
bool HypothesisList_ObjectHypothesis::has_x() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
void HypothesisList_ObjectHypothesis::set_has_x() {
  _has_bits_[0] |= 0x00000001u;
}
void HypothesisList_ObjectHypothesis::clear_has_x() {
  _has_bits_[0] &= ~0x00000001u;
}
void HypothesisList_ObjectHypothesis::clear_x() {
  x_ = 0;
  clear_has_x();
}
float HypothesisList_ObjectHypothesis::x() const {
  // @@protoc_insertion_point(field_get:HypothesisList.ObjectHypothesis.x)
  return x_;
}
void HypothesisList_ObjectHypothesis::set_x(float value) {
  set_has_x();
  x_ = value;
  // @@protoc_insertion_point(field_set:HypothesisList.ObjectHypothesis.x)
}

// optional float y = 2;
bool HypothesisList_ObjectHypothesis::has_y() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
void HypothesisList_ObjectHypothesis::set_has_y() {
  _has_bits_[0] |= 0x00000002u;
}
void HypothesisList_ObjectHypothesis::clear_has_y() {
  _has_bits_[0] &= ~0x00000002u;
}
void HypothesisList_ObjectHypothesis::clear_y() {
  y_ = 0;
  clear_has_y();
}
float HypothesisList_ObjectHypothesis::y() const {
  // @@protoc_insertion_point(field_get:HypothesisList.ObjectHypothesis.y)
  return y_;
}
void HypothesisList_ObjectHypothesis::set_y(float value) {
  set_has_y();
  y_ = value;
  // @@protoc_insertion_point(field_set:HypothesisList.ObjectHypothesis.y)
}

// optional float scale = 3;
bool HypothesisList_ObjectHypothesis::has_scale() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
void HypothesisList_ObjectHypothesis::set_has_scale() {
  _has_bits_[0] |= 0x00000004u;
}
void HypothesisList_ObjectHypothesis::clear_has_scale() {
  _has_bits_[0] &= ~0x00000004u;
}
void HypothesisList_ObjectHypothesis::clear_scale() {
  scale_ = 0;
  clear_has_scale();
}
float HypothesisList_ObjectHypothesis::scale() const {
  // @@protoc_insertion_point(field_get:HypothesisList.ObjectHypothesis.scale)
  return scale_;
}
void HypothesisList_ObjectHypothesis::set_scale(float value) {
  set_has_scale();
  scale_ = value;
  // @@protoc_insertion_point(field_set:HypothesisList.ObjectHypothesis.scale)
}

// optional float score = 4;
bool HypothesisList_ObjectHypothesis::has_score() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
void HypothesisList_ObjectHypothesis::set_has_score() {
  _has_bits_[0] |= 0x00000008u;
}
void HypothesisList_ObjectHypothesis::clear_has_score() {
  _has_bits_[0] &= ~0x00000008u;
}
void HypothesisList_ObjectHypothesis::clear_score() {
  score_ = 0;
  clear_has_score();
}
float HypothesisList_ObjectHypothesis::score() const {
  // @@protoc_insertion_point(field_get:HypothesisList.ObjectHypothesis.score)
  return score_;
}
void HypothesisList_ObjectHypothesis::set_score(float value) {
  set_has_score();
  score_ = value;
  // @@protoc_insertion_point(field_set:HypothesisList.ObjectHypothesis.score)
}

// optional bool flip = 5 [default = false];
bool HypothesisList_ObjectHypothesis::has_flip() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
void HypothesisList_ObjectHypothesis::set_has_flip() {
  _has_bits_[0] |= 0x00000010u;
}
void HypothesisList_ObjectHypothesis::clear_has_flip() {
  _has_bits_[0] &= ~0x00000010u;
}
void HypothesisList_ObjectHypothesis::clear_flip() {
  flip_ = false;
  clear_has_flip();
}
bool HypothesisList_ObjectHypothesis::flip() const {
  // @@protoc_insertion_point(field_get:HypothesisList.ObjectHypothesis.flip)
  return flip_;
}
void HypothesisList_ObjectHypothesis::set_flip(bool value) {
  set_has_flip();
  flip_ = value;
  // @@protoc_insertion_point(field_set:HypothesisList.ObjectHypothesis.flip)
}

inline const HypothesisList_ObjectHypothesis* HypothesisList_ObjectHypothesis::internal_default_instance() {
  return &HypothesisList_ObjectHypothesis_default_instance_.get();
}
// -------------------------------------------------------------------

// HypothesisList

// repeated .HypothesisList.ObjectHypothesis hyp = 1;
int HypothesisList::hyp_size() const {
  return hyp_.size();
}
void HypothesisList::clear_hyp() {
  hyp_.Clear();
}
const ::HypothesisList_ObjectHypothesis& HypothesisList::hyp(int index) const {
  // @@protoc_insertion_point(field_get:HypothesisList.hyp)
  return hyp_.Get(index);
}
::HypothesisList_ObjectHypothesis* HypothesisList::mutable_hyp(int index) {
  // @@protoc_insertion_point(field_mutable:HypothesisList.hyp)
  return hyp_.Mutable(index);
}
::HypothesisList_ObjectHypothesis* HypothesisList::add_hyp() {
  // @@protoc_insertion_point(field_add:HypothesisList.hyp)
  return hyp_.Add();
}
::google::protobuf::RepeatedPtrField< ::HypothesisList_ObjectHypothesis >*
HypothesisList::mutable_hyp() {
  // @@protoc_insertion_point(field_mutable_list:HypothesisList.hyp)
  return &hyp_;
}
const ::google::protobuf::RepeatedPtrField< ::HypothesisList_ObjectHypothesis >&
HypothesisList::hyp() const {
  // @@protoc_insertion_point(field_list:HypothesisList.hyp)
  return hyp_;
}

inline const HypothesisList* HypothesisList::internal_default_instance() {
  return &HypothesisList_default_instance_.get();
}
#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

// @@protoc_insertion_point(global_scope)
