#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace uni20::test
{

class EnvVarGuard {
  public:
    explicit EnvVarGuard(std::string name) : name_(std::move(name))
    {
      if (char const* value = std::getenv(name_.c_str()))
      {
        original_ = value;
      }
    }

    EnvVarGuard(std::string name, std::string value) : EnvVarGuard(std::move(name)) { set(value); }

    EnvVarGuard(EnvVarGuard const&) = delete;
    EnvVarGuard& operator=(EnvVarGuard const&) = delete;

    ~EnvVarGuard()
    {
      if (original_)
      {
        ::setenv(name_.c_str(), original_->c_str(), 1);
      }
      else
      {
        ::unsetenv(name_.c_str());
      }
    }

    void set(std::string const& value) const { ::setenv(name_.c_str(), value.c_str(), 1); }

    void unset() const { ::unsetenv(name_.c_str()); }

  private:
    std::string name_;
    std::optional<std::string> original_;
};

} // namespace uni20::test
