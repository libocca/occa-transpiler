#pragma once

#include <nlohmann/json.hpp>

#include <list>
#include <optional>
#include <string>
#include <vector>

namespace oklt {

enum struct DatatypeCategory {
    BUILTIN,
    CUSTOM,
};

//// map TaskState values to JSON as strings
// NLOHMANN_JSON_SERIALIZE_ENUM(
//   DatatypeCategory, {
//   {DatatypeCategory::BUILTIN, "builtin"},
//   {DatatypeCategory::CUSTOM, "custom"},
// });

struct DataType {
    std::string name;
    DatatypeCategory type;
    int bytes;
};

struct ArgumentInfo {
    bool is_const;
    DataType dtype;
    std::string name;
    bool is_ptr;
};

enum class LoopMetaType { Regular, Outer, Inner, OuterInner };
inline std::string toString(LoopMetaType lmd) {
    switch (lmd) {
        case LoopMetaType::Regular:
            return "regular";
        case LoopMetaType::OuterInner:
            return "outer_inner";
        case LoopMetaType::Outer:
            return "outer";
        case LoopMetaType::Inner:
            return "inner";
    }
    return "<uknown>";
}

// TODO replace clang types by own
enum class BinOp { Eq, Le, Lt, Gt, Ge, AddAssign, RemoveAssign, Other };

enum class UnOp { PreInc, PostInc, PreDec, PostDec, Other };

struct LoopMetaData {
    LoopMetaType type = LoopMetaType::Regular;
    std::list<LoopMetaData> childrens;

    struct {
        std::string type;
        std::string name;
    } var;
    struct {
        std::string start;
        std::string end;
        size_t size = 0;
    } range;
    struct {
        std::string cmp;
        BinOp op = BinOp::Eq;
    } condition;
    struct {
        std::string val;
        union {
            UnOp uo;
            BinOp bo;
        } op;
    } inc;

    [[nodiscard]] bool IsInc() const {
        bool ret = false;
        if (inc.val.empty()) {
            ret = (inc.op.uo == UnOp::PreInc || inc.op.uo == UnOp::PostInc);
        } else {
            ret = (inc.op.bo == BinOp::AddAssign);
        }

        ret = (ret && (condition.op == BinOp::Le || condition.op == BinOp::Lt));

        return ret;
    };
    [[nodiscard]] bool isUnary() const {
        if (!inc.val.empty()) {
            return false;
        }
        // should by unnecessary check, but just in case
        return (inc.op.uo == UnOp::PreInc) || (inc.op.uo == UnOp::PostInc) ||
               (inc.op.uo == UnOp::PreDec) || (inc.op.uo == UnOp::PostDec);
    };

    [[nodiscard]] std::string getRangeSizeStr() const {
        if (IsInc()) {
            return range.end + " - " + range.start;
        } else {
            return range.start + " - " + range.end;
        };
    };

    [[nodiscard]] bool isOuter() const {
        return type == LoopMetaType::Outer || type == LoopMetaType::OuterInner;
    };
    [[nodiscard]] bool isInner() const {
        return type == LoopMetaType::Inner || type == LoopMetaType::OuterInner;
    };
    [[nodiscard]] bool isRegular() const { return type == LoopMetaType::Regular; };
};

struct KernelInfo {
    std::string name;
    std::vector<ArgumentInfo> args;
    std::list<LoopMetaData> childrens;
};

struct DependeciesInfo {};

struct PropertyInfo {
    std::string compiler;
    std::string compiler_flags;
    std::string hash;
    std::string mode;
    bool verbose;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PropertyInfo, compiler, compiler_flags, hash, mode, verbose);
};

struct ProgramMetaData {
    std::optional<DependeciesInfo> dependencies = std::nullopt;
    std::string hash;
    std::vector<KernelInfo> kernels;
    std::optional<PropertyInfo> props = std::nullopt;
    KernelInfo& addKernelInfo(std::string name, size_t numArg, size_t numKern) {
        // create new slot
        // there is no way to pass args nicely for POD
        kernels.emplace_back();

        // get just created slot and fill it
        auto& kiPtr = kernels.back();
        kiPtr.name = std::move(name);
        kiPtr.args.resize(numArg);
        kiPtr.childrens.resize(numKern);

        return kiPtr;
    }
};

// INFO: because of different behaviour of presence field bytes when category is builtin
void to_json(nlohmann::json& json, const DataType& dt);
void from_json(const nlohmann::json& json, DataType& dt);

// INFO: needs custom function because of `const` field name
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

// INFO: skip some fields in serialization/deserialization process
void to_json(nlohmann::json& json, const KernelInfo& kernelMeta);
void from_json(const nlohmann::json& json, KernelInfo& kernelMeta);

// INFO: using optional to be able to have empty json object
void to_json(nlohmann::json& json, const ProgramMetaData& kernelInfo);
void from_json(const nlohmann::json& json, ProgramMetaData& kernelInfo);

}  // namespace oklt
