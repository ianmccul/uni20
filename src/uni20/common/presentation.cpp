#include "presentation.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>

namespace uni20::presentation
{
namespace
{

struct decoded_codepoint
{
    char32_t value = U'\0';
    std::size_t bytes = 1;
    bool valid = false;
};

[[nodiscard]] bool is_continuation(unsigned char byte) { return (byte & 0xC0U) == 0x80U; }

[[nodiscard]] decoded_codepoint decode_next(std::string_view text, std::size_t offset)
{
  auto const remaining = text.size() - offset;
  auto const b0 = static_cast<unsigned char>(text[offset]);

  if (b0 < 0x80U) return {static_cast<char32_t>(b0), 1, true};

  if (b0 >= 0xC2U && b0 <= 0xDFU && remaining >= 2)
  {
    auto const b1 = static_cast<unsigned char>(text[offset + 1]);
    if (is_continuation(b1)) return {static_cast<char32_t>(((b0 & 0x1FU) << 6U) | (b1 & 0x3FU)), 2, true};
  }

  if (b0 >= 0xE0U && b0 <= 0xEFU && remaining >= 3)
  {
    auto const b1 = static_cast<unsigned char>(text[offset + 1]);
    auto const b2 = static_cast<unsigned char>(text[offset + 2]);
    bool const valid_prefix =
        (b0 == 0xE0U && b1 >= 0xA0U && b1 <= 0xBFU) || (b0 >= 0xE1U && b0 <= 0xECU && is_continuation(b1)) ||
        (b0 == 0xEDU && b1 >= 0x80U && b1 <= 0x9FU) || (b0 >= 0xEEU && b0 <= 0xEFU && is_continuation(b1));
    if (valid_prefix && is_continuation(b2))
    {
      char32_t const value = static_cast<char32_t>(((b0 & 0x0FU) << 12U) | ((b1 & 0x3FU) << 6U) | (b2 & 0x3FU));
      return {value, 3, true};
    }
  }

  if (b0 >= 0xF0U && b0 <= 0xF4U && remaining >= 4)
  {
    auto const b1 = static_cast<unsigned char>(text[offset + 1]);
    auto const b2 = static_cast<unsigned char>(text[offset + 2]);
    auto const b3 = static_cast<unsigned char>(text[offset + 3]);
    bool const valid_prefix = (b0 == 0xF0U && b1 >= 0x90U && b1 <= 0xBFU) ||
                              (b0 >= 0xF1U && b0 <= 0xF3U && is_continuation(b1)) ||
                              (b0 == 0xF4U && b1 >= 0x80U && b1 <= 0x8FU);
    if (valid_prefix && is_continuation(b2) && is_continuation(b3))
    {
      char32_t const value =
          static_cast<char32_t>(((b0 & 0x07U) << 18U) | ((b1 & 0x3FU) << 12U) | ((b2 & 0x3FU) << 6U) | (b3 & 0x3FU));
      return {value, 4, true};
    }
  }

  return {static_cast<char32_t>(b0), 1, false};
}

void append_utf8(std::string& out, char32_t value)
{
  if (value <= 0x7FU)
  {
    out.push_back(static_cast<char>(value));
  }
  else if (value <= 0x7FFU)
  {
    out.push_back(static_cast<char>(0xC0U | ((value >> 6U) & 0x1FU)));
    out.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
  else if (value <= 0xFFFFU)
  {
    out.push_back(static_cast<char>(0xE0U | ((value >> 12U) & 0x0FU)));
    out.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
  else
  {
    out.push_back(static_cast<char>(0xF0U | ((value >> 18U) & 0x07U)));
    out.push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
}

void append_byte_escape(std::string& out, unsigned char byte)
{
  char buffer[5] = {};
  std::snprintf(buffer, sizeof(buffer), "\\x%02X", static_cast<unsigned>(byte));
  out += buffer;
}

void append_codepoint_escape(std::string& out, char32_t value)
{
  char buffer[11] = {};
  if (value <= 0xFFFFU)
  {
    std::snprintf(buffer, sizeof(buffer), "\\u%04X", static_cast<unsigned>(value));
  }
  else
  {
    std::snprintf(buffer, sizeof(buffer), "\\U%08X", static_cast<unsigned>(value));
  }
  out += buffer;
}

[[nodiscard]] terminal::TerminalStyle style(std::string_view spec) { return terminal::TerminalStyle(spec); }

[[nodiscard]] terminal::TerminalStyle status_style(semantic_glyph glyph)
{
  switch (glyph)
  {
    case semantic_glyph::success:
      return style("Green;Bold");
    case semantic_glyph::failure:
      return style("Red;Bold");
    case semantic_glyph::warning:
      return style("Yellow;Bold");
    case semantic_glyph::info:
      return style("LightBlue;Bold");
    default:
      return style("Bold");
  }
}

[[nodiscard]] bool in_range(char32_t value, char32_t lo, char32_t hi) { return value >= lo && value <= hi; }

[[nodiscard]] bool is_combining(char32_t value)
{
  return in_range(value, 0x0300, 0x036F) || in_range(value, 0x1AB0, 0x1AFF) || in_range(value, 0x1DC0, 0x1DFF) ||
         in_range(value, 0x20D0, 0x20FF) || in_range(value, 0xFE20, 0xFE2F) || in_range(value, 0xFE00, 0xFE0F);
}

[[nodiscard]] bool is_cjk_wide(char32_t value)
{
  return in_range(value, 0x1100, 0x115F) || value == 0x2329 || value == 0x232A || in_range(value, 0x2E80, 0xA4CF) ||
         in_range(value, 0xAC00, 0xD7A3) || in_range(value, 0xF900, 0xFAFF) || in_range(value, 0xFE10, 0xFE19) ||
         in_range(value, 0xFE30, 0xFE6F) || in_range(value, 0xFF00, 0xFF60) || in_range(value, 0xFFE0, 0xFFE6) ||
         in_range(value, 0x20000, 0x3FFFD);
}

[[nodiscard]] bool is_emoji(char32_t value) { return in_range(value, 0x1F000, 0x1FAFF); }

[[nodiscard]] bool is_emoji_variation_base(char32_t value)
{
  switch (value)
  {
    case 0x203C:
    case 0x2049:
    case 0x2122:
    case 0x2139:
    case 0x2194:
    case 0x2195:
    case 0x2196:
    case 0x2197:
    case 0x2198:
    case 0x2199:
    case 0x21A9:
    case 0x21AA:
    case 0x231A:
    case 0x231B:
    case 0x2328:
    case 0x23CF:
    case 0x23E9:
    case 0x23EA:
    case 0x23EB:
    case 0x23EC:
    case 0x23ED:
    case 0x23EE:
    case 0x23EF:
    case 0x23F0:
    case 0x23F1:
    case 0x23F2:
    case 0x23F3:
    case 0x23F8:
    case 0x23F9:
    case 0x23FA:
    case 0x24C2:
    case 0x25AA:
    case 0x25AB:
    case 0x25B6:
    case 0x25C0:
    case 0x25FB:
    case 0x25FC:
    case 0x25FD:
    case 0x25FE:
    case 0x2600:
    case 0x2601:
    case 0x2602:
    case 0x2603:
    case 0x2604:
    case 0x260E:
    case 0x2611:
    case 0x2614:
    case 0x2615:
    case 0x2618:
    case 0x261D:
    case 0x2620:
    case 0x2622:
    case 0x2623:
    case 0x2626:
    case 0x262A:
    case 0x262E:
    case 0x262F:
    case 0x2638:
    case 0x2639:
    case 0x263A:
    case 0x2640:
    case 0x2642:
    case 0x2648:
    case 0x2649:
    case 0x264A:
    case 0x264B:
    case 0x264C:
    case 0x264D:
    case 0x264E:
    case 0x264F:
    case 0x2650:
    case 0x2651:
    case 0x2652:
    case 0x2653:
    case 0x2660:
    case 0x2663:
    case 0x2665:
    case 0x2666:
    case 0x2668:
    case 0x267B:
    case 0x267E:
    case 0x267F:
    case 0x2692:
    case 0x2693:
    case 0x2694:
    case 0x2695:
    case 0x2696:
    case 0x2697:
    case 0x2699:
    case 0x269B:
    case 0x269C:
    case 0x26A0:
    case 0x26A1:
    case 0x26AA:
    case 0x26AB:
    case 0x26B0:
    case 0x26B1:
    case 0x26BD:
    case 0x26BE:
    case 0x26C4:
    case 0x26C5:
    case 0x26C8:
    case 0x26CE:
    case 0x26CF:
    case 0x26D1:
    case 0x26D3:
    case 0x26D4:
    case 0x26E9:
    case 0x26EA:
    case 0x26F0:
    case 0x26F1:
    case 0x26F2:
    case 0x26F3:
    case 0x26F4:
    case 0x26F5:
    case 0x26F7:
    case 0x26F8:
    case 0x26F9:
    case 0x26FA:
    case 0x26FD:
    case 0x2702:
    case 0x2705:
    case 0x2708:
    case 0x2709:
    case 0x270A:
    case 0x270B:
    case 0x270C:
    case 0x270D:
    case 0x270F:
    case 0x2712:
    case 0x2714:
    case 0x2716:
    case 0x271D:
    case 0x2721:
    case 0x2728:
    case 0x2733:
    case 0x2734:
    case 0x2744:
    case 0x2747:
    case 0x274C:
    case 0x274E:
    case 0x2753:
    case 0x2754:
    case 0x2755:
    case 0x2757:
    case 0x2763:
    case 0x2764:
    case 0x2795:
    case 0x2796:
    case 0x2797:
    case 0x27A1:
    case 0x27B0:
    case 0x27BF:
    case 0x2934:
    case 0x2935:
    case 0x2B05:
    case 0x2B06:
    case 0x2B07:
    case 0x2B1B:
    case 0x2B1C:
    case 0x2B50:
    case 0x2B55:
    case 0x3030:
    case 0x303D:
    case 0x3297:
    case 0x3299:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] bool is_ambiguous(char32_t value)
{
  return value == 0x00A1 || value == 0x00A4 || value == 0x00A7 || value == 0x00A8 || value == 0x00AA ||
         value == 0x00AD || value == 0x00AE || in_range(value, 0x00B0, 0x00B4) || in_range(value, 0x00B6, 0x00BA) ||
         in_range(value, 0x00BC, 0x00BF) || value == 0x00C6 || value == 0x00D0 || value == 0x00D7 || value == 0x00D8 ||
         in_range(value, 0x00DE, 0x00E1) || value == 0x00E6 || in_range(value, 0x00E8, 0x00EA) ||
         in_range(value, 0x00EC, 0x00ED) || value == 0x00F0 || in_range(value, 0x00F2, 0x00F3) || value == 0x00F7 ||
         value == 0x00F8 || value == 0x00FA || value == 0x00FC || value == 0x00FE || value == 0x0101 ||
         value == 0x0111 || value == 0x0113 || value == 0x011B || in_range(value, 0x0126, 0x0127) || value == 0x012B ||
         in_range(value, 0x0131, 0x0133) || value == 0x0138 || in_range(value, 0x013F, 0x0142) || value == 0x0144 ||
         in_range(value, 0x0148, 0x014B) || value == 0x014D || in_range(value, 0x0152, 0x0153) ||
         in_range(value, 0x0166, 0x0167) || value == 0x016B || in_range(value, 0x0391, 0x03A9) ||
         in_range(value, 0x03B1, 0x03C1) || in_range(value, 0x03C3, 0x03C9) || value == 0x0401 ||
         in_range(value, 0x0410, 0x044F) || value == 0x0451 || value == 0x2010 || in_range(value, 0x2013, 0x2016) ||
         in_range(value, 0x2018, 0x2019) || in_range(value, 0x201C, 0x201D) || in_range(value, 0x2020, 0x2022) ||
         in_range(value, 0x2024, 0x2027) || value == 0x2030 || in_range(value, 0x2032, 0x2033) || value == 0x2035 ||
         value == 0x203B || value == 0x203E || value == 0x20AC || value == 0x2103 || value == 0x2105 ||
         value == 0x2109 || value == 0x2113 || value == 0x2116 || in_range(value, 0x2121, 0x2122) || value == 0x2126 ||
         value == 0x212B || in_range(value, 0x2153, 0x2154) || in_range(value, 0x215B, 0x215E) ||
         in_range(value, 0x2160, 0x216B) || in_range(value, 0x2170, 0x2179) || value == 0x2189 ||
         in_range(value, 0x2190, 0x2199) || value == 0x21D2 || value == 0x21D4 || in_range(value, 0x2200, 0x2203) ||
         value == 0x2207 || value == 0x2208 || value == 0x220B || value == 0x220F || value == 0x2211 ||
         value == 0x2215 || value == 0x221A || in_range(value, 0x221D, 0x2220) || value == 0x2223 || value == 0x2225 ||
         in_range(value, 0x2227, 0x222C) || value == 0x222E || in_range(value, 0x2234, 0x2237) ||
         in_range(value, 0x223C, 0x223D) || value == 0x2248 || value == 0x224C || value == 0x2252 ||
         in_range(value, 0x2260, 0x2261) || in_range(value, 0x2264, 0x2267) || in_range(value, 0x226A, 0x226B) ||
         in_range(value, 0x226E, 0x226F) || in_range(value, 0x2282, 0x2283) || in_range(value, 0x2286, 0x2287) ||
         value == 0x2295 || value == 0x2299 || value == 0x22A5 || value == 0x22BF || value == 0x2312 ||
         in_range(value, 0x2460, 0x24E9) || in_range(value, 0x24EB, 0x254B) || in_range(value, 0x2550, 0x2573) ||
         in_range(value, 0x2580, 0x258F) || in_range(value, 0x2592, 0x2595) || in_range(value, 0x25A0, 0x25A1) ||
         in_range(value, 0x25A3, 0x25A9) || in_range(value, 0x25B2, 0x25B3) || in_range(value, 0x25B6, 0x25B7) ||
         in_range(value, 0x25BC, 0x25BD) || in_range(value, 0x25C0, 0x25C1) || in_range(value, 0x25C6, 0x25C8) ||
         value == 0x25CB || in_range(value, 0x25CE, 0x25D1) || in_range(value, 0x25E2, 0x25E5) || value == 0x25EF ||
         in_range(value, 0x2605, 0x2606) || value == 0x2609 || in_range(value, 0x260E, 0x260F) ||
         in_range(value, 0x2614, 0x2615) || value == 0x2640 || value == 0x2642 || in_range(value, 0x2660, 0x2661) ||
         in_range(value, 0x2663, 0x2665) || in_range(value, 0x2667, 0x266A) || in_range(value, 0x266C, 0x266D) ||
         value == 0x266F;
}

[[nodiscard]] std::optional<std::string_view> ascii_symbol_fallback(char32_t value)
{
  switch (value)
  {
    case 0x00A0:
      return " ";
    case 0x00AB:
    case 0x00BB:
    case 0x2018:
    case 0x2019:
    case 0x201A:
    case 0x201B:
      return "'";
    case 0x201C:
    case 0x201D:
    case 0x201E:
    case 0x201F:
      return "\"";
    case 0x2010:
    case 0x2011:
    case 0x2012:
    case 0x2013:
    case 0x2014:
    case 0x2015:
    case 0x2212:
      return "-";
    case 0x2026:
      return "...";
    case 0x2190:
      return "<-";
    case 0x2191:
      return "^";
    case 0x2192:
      return "->";
    case 0x2193:
      return "v";
    case 0x2194:
      return "<->";
    case 0x21D0:
      return "<=";
    case 0x21D2:
      return "=>";
    case 0x21D4:
      return "<=>";
    case 0x2139:
      return "i";
    case 0x25B2:
    case 0x25B3:
    case 0x26A0:
      return "!";
    case 0x2705:
    case 0x2713:
    case 0x2714:
      return "OK";
    case 0x2717:
    case 0x2718:
    case 0x274C:
      return "X";
    case 0x00B1:
      return "+/-";
    case 0x00D7:
      return "x";
    case 0x00F7:
      return "/";
    case 0x2202:
      return "d";
    case 0x2208:
      return "in";
    case 0x2209:
      return "notin";
    case 0x220F:
      return "prod";
    case 0x2211:
      return "sum";
    case 0x221A:
      return "sqrt";
    case 0x221E:
      return "inf";
    case 0x2227:
      return "&&";
    case 0x2228:
      return "||";
    case 0x222B:
      return "int";
    case 0x2248:
      return "~=";
    case 0x2260:
      return "!=";
    case 0x2264:
      return "<=";
    case 0x2265:
      return ">=";
    case 0x25CF:
    case 0x2022:
      return "*";
    case 0x2500:
    case 0x2501:
    case 0x2550:
      return "-";
    case 0x2502:
    case 0x2503:
    case 0x2551:
      return "|";
    case 0x250C:
    case 0x2510:
    case 0x2514:
    case 0x2518:
    case 0x251C:
    case 0x2524:
    case 0x252C:
    case 0x2534:
    case 0x253C:
    case 0x2554:
    case 0x2557:
    case 0x255A:
    case 0x255D:
    case 0x2560:
    case 0x2563:
    case 0x2566:
    case 0x2569:
    case 0x256C:
    case 0x256D:
    case 0x256E:
    case 0x256F:
    case 0x2570:
      return "+";
    case 0x2571:
      return "/";
    case 0x2572:
      return "\\";
    case 0x2573:
      return "x";
    case 0xFE0E:
    case 0xFE0F:
      return "";
    default:
      return std::nullopt;
  }
}

[[nodiscard]] bool is_ornament(semantic_glyph glyph)
{
  switch (glyph)
  {
    case semantic_glyph::box_horizontal:
    case semantic_glyph::box_vertical:
    case semantic_glyph::box_top_left:
    case semantic_glyph::box_top_right:
    case semantic_glyph::box_bottom_left:
    case semantic_glyph::box_bottom_right:
    case semantic_glyph::box_round_top_left:
    case semantic_glyph::box_round_top_right:
    case semantic_glyph::box_round_bottom_left:
    case semantic_glyph::box_round_bottom_right:
    case semantic_glyph::box_tee_left:
    case semantic_glyph::box_tee_right:
    case semantic_glyph::box_tee_up:
    case semantic_glyph::box_tee_down:
    case semantic_glyph::box_cross:
    case semantic_glyph::box_diagonal_forward:
    case semantic_glyph::box_diagonal_back:
    case semantic_glyph::box_diagonal_cross:
    case semantic_glyph::tree_branch:
    case semantic_glyph::tree_last:
    case semantic_glyph::tree_vertical:
    case semantic_glyph::tree_space:
    case semantic_glyph::bullet:
    case semantic_glyph::matrix_top_left:
    case semantic_glyph::matrix_top_right:
    case semantic_glyph::matrix_middle_left:
    case semantic_glyph::matrix_middle_right:
    case semantic_glyph::matrix_bottom_left:
    case semantic_glyph::matrix_bottom_right:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] std::string ascii_glyph(semantic_glyph glyph)
{
  switch (glyph)
  {
    case semantic_glyph::success:
      return "[OK]";
    case semantic_glyph::failure:
      return "[FAIL]";
    case semantic_glyph::warning:
      return "[WARN]";
    case semantic_glyph::info:
      return "[INFO]";
    case semantic_glyph::arrow_right:
      return "->";
    case semantic_glyph::arrow_left:
      return "<-";
    case semantic_glyph::arrow_up:
      return "^";
    case semantic_glyph::arrow_down:
      return "v";
    case semantic_glyph::arrow_left_right:
      return "<->";
    case semantic_glyph::ellipsis:
      return "...";
    case semantic_glyph::box_horizontal:
      return "-";
    case semantic_glyph::box_vertical:
      return "|";
    case semantic_glyph::box_top_left:
    case semantic_glyph::box_top_right:
    case semantic_glyph::box_bottom_left:
    case semantic_glyph::box_bottom_right:
    case semantic_glyph::box_round_top_left:
    case semantic_glyph::box_round_top_right:
    case semantic_glyph::box_round_bottom_left:
    case semantic_glyph::box_round_bottom_right:
    case semantic_glyph::box_tee_left:
    case semantic_glyph::box_tee_right:
    case semantic_glyph::box_tee_up:
    case semantic_glyph::box_tee_down:
    case semantic_glyph::box_cross:
      return "+";
    case semantic_glyph::box_diagonal_forward:
      return "/";
    case semantic_glyph::box_diagonal_back:
      return "\\";
    case semantic_glyph::box_diagonal_cross:
      return "x";
    case semantic_glyph::tree_branch:
      return "|-";
    case semantic_glyph::tree_last:
      return "`-";
    case semantic_glyph::tree_vertical:
      return "|";
    case semantic_glyph::tree_space:
      return " ";
    case semantic_glyph::bullet:
      return "*";
    case semantic_glyph::matrix_top_left:
    case semantic_glyph::matrix_bottom_left:
      return "[";
    case semantic_glyph::matrix_top_right:
    case semantic_glyph::matrix_bottom_right:
      return "]";
    case semantic_glyph::matrix_middle_left:
    case semantic_glyph::matrix_middle_right:
      return "|";
  }
  return "";
}

[[nodiscard]] std::string unicode_glyph(semantic_glyph glyph)
{
  switch (glyph)
  {
    case semantic_glyph::success:
      return "\xE2\x9C\x93";
    case semantic_glyph::failure:
      return "\xE2\x9C\x97";
    case semantic_glyph::warning:
      return "\xE2\x96\xB2";
    case semantic_glyph::info:
      return "\xE2\x84\xB9";
    case semantic_glyph::arrow_right:
      return "\xE2\x86\x92";
    case semantic_glyph::arrow_left:
      return "\xE2\x86\x90";
    case semantic_glyph::arrow_up:
      return "\xE2\x86\x91";
    case semantic_glyph::arrow_down:
      return "\xE2\x86\x93";
    case semantic_glyph::arrow_left_right:
      return "\xE2\x86\x94";
    case semantic_glyph::ellipsis:
      return "\xE2\x80\xA6";
    case semantic_glyph::box_horizontal:
      return "\xE2\x94\x80";
    case semantic_glyph::box_vertical:
      return "\xE2\x94\x82";
    case semantic_glyph::box_top_left:
      return "\xE2\x94\x8C";
    case semantic_glyph::box_top_right:
      return "\xE2\x94\x90";
    case semantic_glyph::box_bottom_left:
      return "\xE2\x94\x94";
    case semantic_glyph::box_bottom_right:
      return "\xE2\x94\x98";
    case semantic_glyph::box_round_top_left:
      return "\xE2\x95\xAD";
    case semantic_glyph::box_round_top_right:
      return "\xE2\x95\xAE";
    case semantic_glyph::box_round_bottom_left:
      return "\xE2\x95\xB0";
    case semantic_glyph::box_round_bottom_right:
      return "\xE2\x95\xAF";
    case semantic_glyph::box_tee_left:
      return "\xE2\x94\xA4";
    case semantic_glyph::box_tee_right:
      return "\xE2\x94\x9C";
    case semantic_glyph::box_tee_up:
      return "\xE2\x94\xB4";
    case semantic_glyph::box_tee_down:
      return "\xE2\x94\xAC";
    case semantic_glyph::box_cross:
      return "\xE2\x94\xBC";
    case semantic_glyph::box_diagonal_forward:
      return "\xE2\x95\xB1";
    case semantic_glyph::box_diagonal_back:
      return "\xE2\x95\xB2";
    case semantic_glyph::box_diagonal_cross:
      return "\xE2\x95\xB3";
    case semantic_glyph::tree_branch:
      return "\xE2\x94\x9C\xE2\x94\x80";
    case semantic_glyph::tree_last:
      return "\xE2\x94\x94\xE2\x94\x80";
    case semantic_glyph::tree_vertical:
      return "\xE2\x94\x82";
    case semantic_glyph::tree_space:
      return " ";
    case semantic_glyph::bullet:
      return "\xE2\x80\xA2";
    case semantic_glyph::matrix_top_left:
      return "\xE2\x8E\xA1";
    case semantic_glyph::matrix_top_right:
      return "\xE2\x8E\xA4";
    case semantic_glyph::matrix_middle_left:
      return "\xE2\x8E\xA2";
    case semantic_glyph::matrix_middle_right:
      return "\xE2\x8E\xA5";
    case semantic_glyph::matrix_bottom_left:
      return "\xE2\x8E\xA3";
    case semantic_glyph::matrix_bottom_right:
      return "\xE2\x8E\xA6";
  }
  return "";
}

[[nodiscard]] std::string emoji_glyph(semantic_glyph glyph)
{
  switch (glyph)
  {
    case semantic_glyph::success:
      return "\xE2\x9C\x85";
    case semantic_glyph::failure:
      return "\xE2\x9D\x8C";
    case semantic_glyph::warning:
      return "\xF0\x9F\x9A\xA8";
    case semantic_glyph::info:
      return "\xE2\x84\xB9\xEF\xB8\x8F";
    case semantic_glyph::arrow_right:
      return "\xE2\x9E\xA1\xEF\xB8\x8F";
    case semantic_glyph::arrow_left:
      return "\xE2\xAC\x85\xEF\xB8\x8F";
    case semantic_glyph::arrow_up:
      return "\xE2\xAC\x86\xEF\xB8\x8F";
    case semantic_glyph::arrow_down:
      return "\xE2\xAC\x87\xEF\xB8\x8F";
    case semantic_glyph::arrow_left_right:
      return "\xE2\x86\x94\xEF\xB8\x8F";
    default:
      return unicode_glyph(glyph);
  }
}

[[nodiscard]] bool has_style(terminal::TerminalStyle const& style)
{
  return style.fg.has_value() || style.bg.has_value() ||
         static_cast<unsigned>(style.attrs) != static_cast<unsigned>(terminal::ColorAttribute::None);
}

[[nodiscard]] bool ansi_sequence_at(std::string_view text, std::size_t offset, std::size_t& length)
{
  if (static_cast<unsigned char>(text[offset]) != 0x1BU)
  {
    length = 0;
    return false;
  }

  if (offset + 1 >= text.size() || text[offset + 1] != '[')
  {
    length = 1;
    return true;
  }

  std::size_t i = offset + 2;
  while (i < text.size())
  {
    auto const byte = static_cast<unsigned char>(text[i]);
    if (byte >= 0x40U && byte <= 0x7EU)
    {
      length = i - offset + 1;
      return true;
    }
    ++i;
  }

  length = text.size() - offset;
  return true;
}

[[nodiscard]] std::size_t codepoint_width(char32_t value, output_policy const& policy)
{
  if (value < 0x20U || (value >= 0x7FU && value < 0xA0U)) return 0;
  if (is_combining(value)) return 0;
  if (is_emoji(value)) return 2;
  if (is_cjk_wide(value)) return 2;
  if (is_ambiguous(value)) return policy.ambiguous == ambiguous_width::wide ? 2 : 1;
  return 1;
}

[[nodiscard]] std::size_t codepoint_width_at(std::string_view rendered, decoded_codepoint const& decoded,
                                             std::size_t next_offset, output_policy const& policy)
{
  if (decoded.valid && is_emoji_variation_base(decoded.value) && next_offset < rendered.size())
  {
    auto const next = decode_next(rendered, next_offset);
    if (next.valid && next.value == 0xFE0F) return 2;
  }
  return decoded.valid ? codepoint_width(decoded.value, policy) : 1;
}

[[nodiscard]] std::size_t tab_advance(std::size_t column, std::size_t tab_width)
{
  std::size_t const effective_tab_width = std::max<std::size_t>(tab_width, 1);
  return effective_tab_width - (column % effective_tab_width);
}

[[nodiscard]] std::size_t bytes_without_ansi(std::string_view rendered)
{
  std::size_t width = 0;
  for (std::size_t offset = 0; offset < rendered.size();)
  {
    std::size_t ansi_length = 0;
    if (ansi_sequence_at(rendered, offset, ansi_length))
    {
      offset += ansi_length;
      continue;
    }
    auto const decoded = decode_next(rendered, offset);
    width += decoded.bytes;
    offset += decoded.bytes;
  }
  return width;
}

[[nodiscard]] std::size_t display_cells_of_rendered(std::string_view rendered, output_policy const& policy,
                                                    std::size_t initial_column)
{
  std::size_t current = initial_column;
  std::size_t max_column = initial_column;

  for (std::size_t offset = 0; offset < rendered.size();)
  {
    std::size_t ansi_length = 0;
    if (ansi_sequence_at(rendered, offset, ansi_length))
    {
      offset += ansi_length;
      continue;
    }

    auto const decoded = decode_next(rendered, offset);
    offset += decoded.bytes;

    if (!decoded.valid)
    {
      ++current;
      max_column = std::max(max_column, current);
      continue;
    }

    if (decoded.value == U'\n')
    {
      max_column = std::max(max_column, current);
      current = 0;
      continue;
    }

    if (decoded.value == U'\r')
    {
      current = 0;
      continue;
    }

    if (decoded.value == U'\t')
    {
      current += tab_advance(current, policy.tab_width);
      max_column = std::max(max_column, current);
      continue;
    }

    current += codepoint_width_at(rendered, decoded, offset, policy);
    max_column = std::max(max_column, current);
  }

  return max_column >= initial_column ? max_column - initial_column : max_column;
}

[[nodiscard]] std::size_t width_of_rendered(std::string_view rendered, output_policy const& policy,
                                            std::size_t initial_column = 0)
{
  if (policy.width == width_mode::bytes) return bytes_without_ansi(rendered);
  return display_cells_of_rendered(rendered, policy, initial_column);
}

[[nodiscard]] std::string render_span_text(std::string_view text, terminal::TerminalStyle const& style,
                                           output_policy const& policy)
{
  auto rendered = render_text(text, policy);
  if (!should_emit_color(policy) || !has_style(style) || rendered.empty()) return rendered;
  return style.to_string() + rendered + "\033[0m";
}

[[nodiscard]] std::string clip_rendered_to_width(std::string_view rendered, std::size_t max_width,
                                                 output_policy const& policy, std::size_t initial_column)
{
  if (width_of_rendered(rendered, policy, initial_column) <= max_width) return std::string(rendered);

  std::string out;
  std::size_t used = 0;
  std::size_t column = initial_column;

  for (std::size_t offset = 0; offset < rendered.size();)
  {
    std::size_t ansi_length = 0;
    if (ansi_sequence_at(rendered, offset, ansi_length))
    {
      out.append(rendered.substr(offset, ansi_length));
      offset += ansi_length;
      continue;
    }

    auto const decoded = decode_next(rendered, offset);
    std::size_t unit_width = decoded.bytes;
    if (policy.width == width_mode::display_cells)
    {
      if (!decoded.valid)
      {
        unit_width = 1;
      }
      else if (decoded.value == U'\t')
      {
        unit_width = tab_advance(column, policy.tab_width);
      }
      else if (decoded.value == U'\n' || decoded.value == U'\r')
      {
        unit_width = 0;
      }
      else
      {
        unit_width = codepoint_width_at(rendered, decoded, offset + decoded.bytes, policy);
      }
    }

    if (used + unit_width > max_width) break;

    out.append(rendered.substr(offset, decoded.bytes));
    used += unit_width;
    if (policy.width == width_mode::display_cells)
    {
      if (decoded.valid && decoded.value == U'\n')
      {
        column = 0;
      }
      else if (decoded.valid && decoded.value == U'\r')
      {
        column = 0;
      }
      else
      {
        column += unit_width;
      }
    }

    offset += decoded.bytes;
  }

  return out;
}

[[nodiscard]] std::vector<std::size_t> rendered_unit_offsets(std::string_view rendered)
{
  std::vector<std::size_t> offsets;
  offsets.reserve(rendered.size() + 1);
  for (std::size_t offset = 0; offset < rendered.size();)
  {
    offsets.push_back(offset);

    std::size_t ansi_length = 0;
    if (ansi_sequence_at(rendered, offset, ansi_length))
    {
      offset += ansi_length;
      continue;
    }

    auto const decoded = decode_next(rendered, offset);
    offset += decoded.bytes;
  }

  offsets.push_back(rendered.size());
  return offsets;
}

[[nodiscard]] std::string truncate_rendered_left_to_width(std::string_view rendered, std::size_t max_width,
                                                          output_policy const& policy, std::string_view rendered_marker,
                                                          std::size_t initial_column)
{
  if (width_of_rendered(rendered, policy, initial_column) <= max_width) return std::string(rendered);

  auto const marker_width = width_of_rendered(rendered_marker, policy, initial_column);
  if (!rendered_marker.empty() && marker_width >= max_width)
  {
    return clip_rendered_to_width(rendered_marker, max_width, policy, initial_column);
  }

  auto const offsets = rendered_unit_offsets(rendered);
  for (auto const start : offsets)
  {
    std::string candidate;
    candidate.reserve(rendered_marker.size() + rendered.size() - start);
    candidate += rendered_marker;
    candidate += rendered.substr(start);
    if (width_of_rendered(candidate, policy, initial_column) <= max_width) return candidate;
  }

  return std::string(rendered_marker);
}

[[nodiscard]] glyph_set parse_glyph_set(std::string_view value, glyph_set fallback)
{
  if (iequals(value, "unicode")) return glyph_set::unicode;
  if (iequals(value, "emoji")) return glyph_set::emoji;
  if (iequals(value, "ascii")) return glyph_set::ascii;
  return fallback;
}

[[nodiscard]] text_charset parse_text_charset(std::string_view value, text_charset fallback)
{
  if (iequals(value, "utf8") || iequals(value, "utf-8")) return text_charset::utf8;
  if (iequals(value, "ascii_escape") || iequals(value, "ascii-escape") || iequals(value, "escape"))
    return text_charset::ascii_escape;
  if (iequals(value, "ascii_replace") || iequals(value, "ascii-replace") || iequals(value, "replace"))
    return text_charset::ascii_replace;
  return fallback;
}

[[nodiscard]] color_mode parse_color_mode(std::string_view value, color_mode fallback)
{
  if (iequals(value, "auto") || iequals(value, "automatic")) return color_mode::automatic;
  if (iequals(value, "yes") || iequals(value, "always") || iequals(value, "true") || iequals(value, "on") ||
      iequals(value, "1"))
    return color_mode::always;
  if (iequals(value, "no") || iequals(value, "never") || iequals(value, "false") || iequals(value, "off") ||
      iequals(value, "0"))
    return color_mode::never;
  return fallback;
}

void apply_global_environment(output_policy& policy)
{
  if (auto const* raw = std::getenv("UNI20_GLYPHS"))
  {
    policy.glyphs = parse_glyph_set(raw, policy.glyphs);
  }

  if (auto const* raw = std::getenv("UNI20_CHARSET"))
  {
    policy.charset = parse_text_charset(raw, policy.charset);
  }

  if (auto const* raw = std::getenv("UNI20_COLOR"))
  {
    policy.color = parse_color_mode(raw, policy.color);
  }
}

} // namespace

styled_text& styled_text::append(std::string_view text, terminal::TerminalStyle style)
{
  spans_.push_back(styled_text_span{std::string(text), std::move(style)});
  return *this;
}

styled_text& styled_text::append(semantic_glyph glyph, terminal::TerminalStyle style)
{
  spans_.push_back(semantic_glyph_span{glyph, std::move(style)});
  return *this;
}

styled_text& styled_text::append(styled_text const& other)
{
  spans_.insert(spans_.end(), other.spans_.begin(), other.spans_.end());
  return *this;
}

bool styled_text::empty() const noexcept { return spans_.empty(); }

std::vector<presentation_span> const& styled_text::spans() const noexcept { return spans_; }

output_policy terminal_policy(std::FILE* stream)
{
  output_policy policy;
  policy.color = color_mode::automatic;
  policy.output_stream = stream;
  apply_global_environment(policy);
  return policy;
}

output_policy plain_policy()
{
  output_policy policy;
  policy.color = color_mode::never;
  policy.output_stream = nullptr;
  return policy;
}

output_policy strict_ascii_policy(text_charset charset)
{
  output_policy policy = plain_policy();
  policy.glyphs = glyph_set::ascii;
  policy.charset = charset;
  return policy;
}

bool should_emit_color(output_policy const& policy)
{
  if (policy.color == color_mode::never) return false;
  if (policy.color == color_mode::always) return true;
  return !terminal::no_color_requested() && terminal::is_a_terminal(policy.output_stream);
}

std::string render_glyph(semantic_glyph glyph, output_policy const& policy)
{
  if (policy.ornaments == ornament_mode::none && is_ornament(glyph)) return "";

  if (policy.glyphs == glyph_set::ascii || (policy.ornaments == ornament_mode::minimal && is_ornament(glyph)))
  {
    return ascii_glyph(glyph);
  }

  if (policy.glyphs == glyph_set::emoji) return emoji_glyph(glyph);
  return unicode_glyph(glyph);
}

std::string render_text(std::string_view text, output_policy const& policy)
{
  std::string out;
  out.reserve(text.size());

  for (std::size_t offset = 0; offset < text.size();)
  {
    auto const decoded = decode_next(text, offset);
    if (!decoded.valid)
    {
      if (policy.invalid == invalid_utf8::escape)
      {
        append_byte_escape(out, static_cast<unsigned char>(text[offset]));
      }
      else if (policy.charset == text_charset::utf8)
      {
        append_utf8(out, 0xFFFDU);
      }
      else
      {
        out.push_back('?');
      }
      offset += decoded.bytes;
      continue;
    }

    if (decoded.value < 0x80U)
    {
      out.push_back(static_cast<char>(decoded.value));
      offset += decoded.bytes;
      continue;
    }

    if (policy.charset == text_charset::utf8)
    {
      append_utf8(out, decoded.value);
      offset += decoded.bytes;
      continue;
    }

    if (auto fallback = ascii_symbol_fallback(decoded.value); fallback.has_value())
    {
      out += *fallback;
    }
    else if (policy.charset == text_charset::ascii_escape)
    {
      append_codepoint_escape(out, decoded.value);
    }
    else
    {
      out.push_back('?');
    }
    offset += decoded.bytes;
  }

  return out;
}

std::string render(styled_text const& text, output_policy const& policy)
{
  std::string out;
  for (auto const& span : text.spans())
  {
    if (auto const* text_span = std::get_if<styled_text_span>(&span))
    {
      out += render_span_text(text_span->text, text_span->style, policy);
    }
    else if (auto const* glyph_span = std::get_if<semantic_glyph_span>(&span))
    {
      out += render_span_text(render_glyph(glyph_span->glyph, policy), glyph_span->style, policy);
    }
  }

  if (policy.wrap_width.has_value())
  {
    auto const lines = wrap_text(out, *policy.wrap_width, policy);
    std::string wrapped;
    for (std::size_t i = 0; i < lines.size(); ++i)
    {
      if (i > 0) wrapped.push_back('\n');
      wrapped += lines[i];
    }
    return wrapped;
  }

  return out;
}

std::string render_terminal(styled_text const& text, output_policy policy, std::FILE* stream)
{
  policy.output_stream = stream;
  return render(text, policy);
}

std::string render_plain(styled_text const& text, output_policy policy)
{
  policy.color = color_mode::never;
  return render(text, policy);
}

std::string render_strict_ascii(styled_text const& text, text_charset charset, output_policy policy)
{
  policy.color = color_mode::never;
  policy.glyphs = glyph_set::ascii;
  policy.charset = charset;
  return render(text, policy);
}

report_table::report_table(std::string title) : title_(std::move(title)) {}

report_table& report_table::column(std::string heading, table_alignment alignment)
{
  columns_.push_back(table_column{std::move(heading), alignment});
  return *this;
}

report_table& report_table::row(std::vector<std::string> cells)
{
  rows_.push_back(std::move(cells));
  return *this;
}

report_table& report_table::borders(table_border_options options)
{
  border_options_ = options;
  return *this;
}

report_table& report_table::outer_border(bool enabled)
{
  border_options_.outer = enabled;
  return *this;
}

report_table& report_table::column_separators(bool enabled)
{
  border_options_.column_separators = enabled;
  return *this;
}

report_table& report_table::row_separators(bool enabled)
{
  border_options_.row_separators = enabled;
  return *this;
}

report_table& report_table::header_separator(bool enabled)
{
  border_options_.header_separator = enabled;
  return *this;
}

report_table& report_table::grid(bool enabled)
{
  border_options_.outer = enabled;
  border_options_.column_separators = enabled;
  border_options_.row_separators = enabled;
  border_options_.header_separator = enabled;
  return *this;
}

std::string const& report_table::title() const noexcept { return title_; }

std::vector<table_column> const& report_table::columns() const noexcept { return columns_; }

std::vector<std::vector<std::string>> const& report_table::rows() const noexcept { return rows_; }

table_border_options const& report_table::border_options() const noexcept { return border_options_; }

report_builder::report_builder(std::string title) : title_(std::move(title)) {}

report_builder& report_builder::status(semantic_glyph glyph, std::string label)
{
  statuses_.push_back({glyph, std::move(label)});
  return *this;
}

report_builder& report_builder::field(std::string key, std::string value)
{
  fields_.push_back({std::move(key), std::move(value)});
  return *this;
}

report_table& report_builder::table(std::string title)
{
  tables_.push_back(report_table(std::move(title)));
  return tables_.back();
}

std::string const& report_builder::title() const noexcept { return title_; }

std::vector<std::pair<semantic_glyph, std::string>> const& report_builder::statuses() const noexcept
{
  return statuses_;
}

std::vector<std::pair<std::string, std::string>> const& report_builder::fields() const noexcept { return fields_; }

std::vector<report_table> const& report_builder::tables() const noexcept { return tables_; }

namespace
{

[[nodiscard]] std::string pad_table_cell(std::string_view text, std::size_t width, table_alignment alignment,
                                         output_policy const& policy)
{
  switch (alignment)
  {
    case table_alignment::left:
      return pad_right(text, width, policy);
    case table_alignment::right:
      return pad_left(text, width, policy);
    case table_alignment::center:
      return pad_center(text, width, policy);
  }
  return pad_right(text, width, policy);
}

[[nodiscard]] std::size_t table_column_count(report_table const& table)
{
  std::size_t count = table.columns().size();
  for (auto const& row : table.rows())
  {
    count = std::max(count, row.size());
  }
  return count;
}

[[nodiscard]] table_alignment column_alignment(report_table const& table, std::size_t column)
{
  if (column < table.columns().size()) return table.columns()[column].alignment;
  return table_alignment::right;
}

[[nodiscard]] std::string column_heading(report_table const& table, std::size_t column)
{
  if (column < table.columns().size()) return table.columns()[column].heading;
  return {};
}

[[nodiscard]] std::vector<std::size_t> table_widths(report_table const& table, output_policy const& policy)
{
  std::vector<std::size_t> widths(table_column_count(table), 0);
  for (std::size_t i = 0; i < widths.size(); ++i)
  {
    widths[i] = display_width(column_heading(table, i), policy);
  }

  for (auto const& row : table.rows())
  {
    for (std::size_t i = 0; i < row.size(); ++i)
    {
      widths[i] = std::max(widths[i], display_width(row[i], policy));
    }
  }

  return widths;
}

[[nodiscard]] bool is_table_break_space(decoded_codepoint const& decoded)
{
  return decoded.valid && (decoded.value == U' ' || decoded.value == U'\t' || decoded.value == U'\n');
}

[[nodiscard]] std::size_t longest_unbreakable_width(std::string_view text, output_policy const& policy)
{
  auto const rendered = render_text(text, policy);
  std::size_t longest = 0;
  std::size_t current = 0;

  for (std::size_t offset = 0; offset < rendered.size();)
  {
    auto const decoded = decode_next(rendered, offset);
    if (is_table_break_space(decoded))
    {
      longest = std::max(longest, current);
      current = 0;
      offset += decoded.bytes;
      continue;
    }

    std::size_t unit_width = decoded.bytes;
    if (policy.width == width_mode::display_cells)
    {
      if (!decoded.valid)
        unit_width = 1;
      else if (decoded.value == U'\r')
        unit_width = 0;
      else
        unit_width = codepoint_width_at(rendered, decoded, offset + decoded.bytes, policy);
    }

    current += unit_width;
    offset += decoded.bytes;
  }

  return std::max(longest, current);
}

[[nodiscard]] std::vector<std::size_t> table_unbreakable_widths(report_table const& table, output_policy const& policy)
{
  std::vector<std::size_t> widths(table_column_count(table), 1);
  for (std::size_t i = 0; i < widths.size(); ++i)
  {
    widths[i] = std::max(widths[i], longest_unbreakable_width(column_heading(table, i), policy));
  }

  for (auto const& row : table.rows())
  {
    for (std::size_t i = 0; i < row.size(); ++i)
    {
      widths[i] = std::max(widths[i], longest_unbreakable_width(row[i], policy));
    }
  }

  return widths;
}

[[nodiscard]] std::size_t compact_table_fixed_width(std::size_t column_count)
{
  if (column_count == 0) return 0;
  return 2 + 2 * (column_count - 1);
}

[[nodiscard]] std::size_t ruled_table_fixed_width(std::size_t column_count, table_border_options const& options)
{
  if (column_count == 0) return 0;
  return 2 + 2 * options.horizontal_padding * column_count + (options.outer ? 2 : 0) +
         (options.column_separators ? column_count - 1 : 2 * (column_count - 1));
}

[[nodiscard]] std::size_t table_fixed_width(std::size_t column_count, table_border_options const& options)
{
  if (options.outer || options.column_separators || options.row_separators || options.header_separator)
  {
    return ruled_table_fixed_width(column_count, options);
  }
  return compact_table_fixed_width(column_count);
}

[[nodiscard]] std::vector<std::size_t> fit_table_widths(std::vector<std::size_t> widths, report_table const& table,
                                                        output_policy const& policy)
{
  if (!policy.wrap_width.has_value() || widths.empty()) return widths;

  auto const fixed_width = table_fixed_width(widths.size(), table.border_options());
  std::size_t const content_budget =
      *policy.wrap_width > fixed_width ? *policy.wrap_width - fixed_width : widths.size();
  if (content_budget >= widths.size())
  {
    auto const unbreakable_widths = table_unbreakable_widths(table, policy);
    while (true)
    {
      auto const total = std::accumulate(widths.begin(), widths.end(), std::size_t{0});
      if (total <= content_budget) return widths;

      std::optional<std::size_t> best_soft_index;
      std::size_t best_soft_width = 0;
      std::optional<std::size_t> widest_index;
      std::size_t widest_width = 0;
      for (std::size_t i = 0; i < widths.size(); ++i)
      {
        if (widths[i] <= 1) continue;
        auto const next_width = widths[i] - 1;
        if (!widest_index.has_value() || widths[i] > widest_width)
        {
          widest_index = i;
          widest_width = widths[i];
        }
        if (unbreakable_widths[i] < widths[i] && next_width >= unbreakable_widths[i] &&
            (!best_soft_index.has_value() || widths[i] > best_soft_width))
        {
          best_soft_index = i;
          best_soft_width = widths[i];
        }
      }

      if (best_soft_index.has_value())
      {
        --widths[*best_soft_index];
        continue;
      }

      if (!widest_index.has_value()) return widths;
      --widths[*widest_index];
    }
  }

  std::fill(widths.begin(), widths.end(), std::size_t{1});
  return widths;
}

[[nodiscard]] bool has_table_rules(table_border_options const& options)
{
  return options.outer || options.column_separators || options.row_separators || options.header_separator;
}

void append_repeated_glyph(styled_text& text, semantic_glyph glyph, std::size_t count,
                           terminal::TerminalStyle line_style)
{
  for (std::size_t i = 0; i < count; ++i)
  {
    text.append(glyph, line_style);
  }
}

[[nodiscard]] std::vector<std::size_t> table_cell_widths(std::vector<std::size_t> const& content_widths,
                                                         table_border_options const& options)
{
  std::vector<std::size_t> widths;
  widths.reserve(content_widths.size());
  for (auto const width : content_widths)
  {
    widths.push_back(width + 2 * options.horizontal_padding);
  }
  return widths;
}

[[nodiscard]] std::string padded_table_cell(std::string_view text, std::size_t width, table_alignment alignment,
                                            table_border_options const& border_options, output_policy const& policy)
{
  return std::string(border_options.horizontal_padding, ' ') + pad_table_cell(text, width, alignment, policy) +
         std::string(border_options.horizontal_padding, ' ');
}

[[nodiscard]] std::vector<std::string> wrapped_table_cell(std::string_view text, std::size_t width,
                                                          output_policy const& policy)
{
  auto lines = wrap_text(text, std::max<std::size_t>(width, 1), policy);
  if (lines.empty()) lines.emplace_back();
  return lines;
}

[[nodiscard]] std::vector<std::vector<std::string>> wrapped_table_cells(std::vector<std::string> const& row,
                                                                        std::vector<std::size_t> const& widths,
                                                                        output_policy const& policy)
{
  std::vector<std::vector<std::string>> cells;
  cells.reserve(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i)
  {
    std::string_view const cell = i < row.size() ? std::string_view(row[i]) : std::string_view{};
    cells.push_back(wrapped_table_cell(cell, widths[i], policy));
  }
  return cells;
}

[[nodiscard]] std::size_t wrapped_row_height(std::vector<std::vector<std::string>> const& cells)
{
  std::size_t height = 1;
  for (auto const& cell : cells)
  {
    height = std::max(height, cell.size());
  }
  return height;
}

void append_table_rule(styled_text& text, std::vector<std::size_t> const& cell_widths,
                       table_border_options const& border_options, semantic_glyph left, semantic_glyph junction,
                       semantic_glyph right)
{
  auto const line_style = style("LightGray");
  text.append("  ");
  if (border_options.outer)
  {
    text.append(left, line_style);
  }

  for (std::size_t i = 0; i < cell_widths.size(); ++i)
  {
    append_repeated_glyph(text, semantic_glyph::box_horizontal, cell_widths[i], line_style);
    if (i + 1 < cell_widths.size())
    {
      if (border_options.column_separators)
      {
        text.append(junction, line_style);
      }
      else
      {
        append_repeated_glyph(text, semantic_glyph::box_horizontal, 2, line_style);
      }
    }
  }

  if (border_options.outer)
  {
    text.append(right, line_style);
  }
  text.append("\n");
}

void append_ruled_table_row(styled_text& text, report_table const& table, std::vector<std::size_t> const& widths,
                            std::vector<std::string> const& row, output_policy const& policy, bool heading = false)
{
  auto const& border_options = table.border_options();
  auto const line_style = style("LightGray");
  auto const wrapped_cells = wrapped_table_cells(row, widths, policy);
  auto const height = wrapped_row_height(wrapped_cells);
  for (std::size_t line = 0; line < height; ++line)
  {
    text.append("  ");
    if (border_options.outer)
    {
      text.append(semantic_glyph::box_vertical, line_style);
    }

    for (std::size_t i = 0; i < widths.size(); ++i)
    {
      if (i > 0)
      {
        if (border_options.column_separators)
        {
          text.append(semantic_glyph::box_vertical, line_style);
        }
        else
        {
          text.append("  ");
        }
      }

      std::string_view const cell =
          line < wrapped_cells[i].size() ? std::string_view(wrapped_cells[i][line]) : std::string_view{};
      text.append(padded_table_cell(cell, widths[i], column_alignment(table, i), border_options, policy),
                  heading ? style("LightGray") : terminal::TerminalStyle{});
    }

    if (border_options.outer)
    {
      text.append(semantic_glyph::box_vertical, line_style);
    }
    text.append("\n");
  }
}

void append_plain_table_row(styled_text& text, report_table const& table, std::vector<std::size_t> const& widths,
                            std::vector<std::string> const& row, output_policy const& policy, bool heading = false)
{
  auto const wrapped_cells = wrapped_table_cells(row, widths, policy);
  auto const height = wrapped_row_height(wrapped_cells);
  for (std::size_t line = 0; line < height; ++line)
  {
    text.append("  ");
    for (std::size_t i = 0; i < widths.size(); ++i)
    {
      if (i > 0) text.append("  ");
      std::string_view const cell =
          line < wrapped_cells[i].size() ? std::string_view(wrapped_cells[i][line]) : std::string_view{};
      text.append(pad_table_cell(cell, widths[i], column_alignment(table, i), policy),
                  heading ? style("LightGray") : terminal::TerminalStyle{});
    }
    text.append("\n");
  }
}

void append_plain_report_table(styled_text& text, report_table const& table, output_policy const& policy,
                               std::vector<std::size_t> const& widths)
{
  std::vector<std::string> headings;
  headings.reserve(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i)
  {
    headings.push_back(column_heading(table, i));
  }
  append_plain_table_row(text, table, widths, headings, policy, true);

  for (auto const& row : table.rows())
  {
    append_plain_table_row(text, table, widths, row, policy);
  }
}

void append_ruled_report_table(styled_text& text, report_table const& table, output_policy const& policy,
                               std::vector<std::size_t> const& widths)
{
  auto const& border_options = table.border_options();
  auto const cell_widths = table_cell_widths(widths, border_options);

  if (border_options.outer)
  {
    append_table_rule(text, cell_widths, border_options, semantic_glyph::box_top_left, semantic_glyph::box_tee_down,
                      semantic_glyph::box_top_right);
  }

  std::vector<std::string> headings;
  headings.reserve(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i)
  {
    headings.push_back(column_heading(table, i));
  }
  append_ruled_table_row(text, table, widths, headings, policy, true);

  if (border_options.header_separator)
  {
    append_table_rule(text, cell_widths, border_options, semantic_glyph::box_tee_right, semantic_glyph::box_cross,
                      semantic_glyph::box_tee_left);
  }

  for (std::size_t row_index = 0; row_index < table.rows().size(); ++row_index)
  {
    append_ruled_table_row(text, table, widths, table.rows()[row_index], policy);
    if (border_options.row_separators && row_index + 1 < table.rows().size())
    {
      append_table_rule(text, cell_widths, border_options, semantic_glyph::box_tee_right, semantic_glyph::box_cross,
                        semantic_glyph::box_tee_left);
    }
  }

  if (border_options.outer)
  {
    append_table_rule(text, cell_widths, border_options, semantic_glyph::box_bottom_left, semantic_glyph::box_tee_up,
                      semantic_glyph::box_bottom_right);
  }
}

void append_report_table(styled_text& text, report_table const& table, output_policy const& policy)
{
  if (!table.title().empty())
  {
    text.append("\n").append(table.title(), style("Cyan")).append("\n");
  }

  auto const widths = fit_table_widths(table_widths(table, policy), table, policy);
  if (widths.empty()) return;

  if (has_table_rules(table.border_options()))
  {
    append_ruled_report_table(text, table, policy, widths);
  }
  else
  {
    append_plain_report_table(text, table, policy, widths);
  }
}

} // namespace

styled_text render_report(report_builder const& report, output_policy const& policy)
{
  styled_text text;
  if (!report.title().empty())
  {
    text.append(report.title(), style("Bold")).append("\n");
  }

  for (auto const& [glyph, label] : report.statuses())
  {
    text.append(glyph, status_style(glyph)).append(" ").append(label, status_style(glyph)).append("\n");
  }

  std::size_t key_width = 0;
  for (auto const& [key, value] : report.fields())
  {
    key_width = std::max(key_width, display_width(key, policy));
  }

  for (auto const& [key, value] : report.fields())
  {
    text.append("  ").append(pad_right(key, key_width + 2, policy), style("LightGray")).append(value).append("\n");
  }

  for (auto const& table : report.tables())
  {
    append_report_table(text, table, policy);
  }

  return text;
}

std::string render_terminal(report_builder const& report, output_policy policy, std::FILE* stream)
{
  policy.output_stream = stream;
  auto render_policy = policy;
  render_policy.wrap_width = std::nullopt;
  return render_terminal(render_report(report, policy), render_policy, stream);
}

std::string render_plain(report_builder const& report, output_policy policy)
{
  auto render_policy = policy;
  render_policy.wrap_width = std::nullopt;
  return render_plain(render_report(report, policy), render_policy);
}

std::size_t display_width(std::string_view text, output_policy const& policy, std::size_t initial_column)
{
  auto const rendered = render_text(text, policy);
  return width_of_rendered(rendered, policy, initial_column);
}

std::size_t display_width(styled_text const& text, output_policy const& policy, std::size_t initial_column)
{
  return width_of_rendered(render_plain(text, policy), policy, initial_column);
}

std::string pad_left(std::string_view text, std::size_t target_width, output_policy const& policy)
{
  auto rendered = render_text(text, policy);
  auto const width = width_of_rendered(rendered, policy);
  if (width >= target_width) return rendered;
  return std::string(target_width - width, ' ') + rendered;
}

std::string pad_right(std::string_view text, std::size_t target_width, output_policy const& policy)
{
  auto rendered = render_text(text, policy);
  auto const width = width_of_rendered(rendered, policy);
  if (width >= target_width) return rendered;
  rendered.append(target_width - width, ' ');
  return rendered;
}

std::string pad_center(std::string_view text, std::size_t target_width, output_policy const& policy)
{
  auto rendered = render_text(text, policy);
  auto const width = width_of_rendered(rendered, policy);
  if (width >= target_width) return rendered;
  auto const total_padding = target_width - width;
  auto const left_padding = total_padding / 2;
  auto const right_padding = total_padding - left_padding;
  return std::string(left_padding, ' ') + rendered + std::string(right_padding, ' ');
}

std::string clip_to_width(std::string_view text, std::size_t max_width, output_policy const& policy,
                          std::size_t initial_column)
{
  return clip_rendered_to_width(render_text(text, policy), max_width, policy, initial_column);
}

std::string truncate_to_width(std::string_view text, std::size_t max_width, output_policy const& policy,
                              std::string_view marker, std::size_t initial_column)
{
  auto const rendered = render_text(text, policy);
  if (width_of_rendered(rendered, policy, initial_column) <= max_width) return rendered;
  if (marker.empty()) return clip_rendered_to_width(rendered, max_width, policy, initial_column);

  auto const rendered_marker = render_text(marker, policy);
  auto const marker_width = width_of_rendered(rendered_marker, policy, initial_column);
  if (marker_width >= max_width) return clip_rendered_to_width(rendered_marker, max_width, policy, initial_column);

  return clip_rendered_to_width(rendered, max_width - marker_width, policy, initial_column) + rendered_marker;
}

std::string truncate_left_to_width(std::string_view text, std::size_t max_width, output_policy const& policy,
                                   std::string_view marker, std::size_t initial_column)
{
  auto const rendered = render_text(text, policy);
  auto const rendered_marker = render_text(marker, policy);
  return truncate_rendered_left_to_width(rendered, max_width, policy, rendered_marker, initial_column);
}

std::string prefix_lines(std::string_view text, std::string_view prefix, output_policy const& policy, bool prefix_first)
{
  auto const rendered = render_text(text, policy);
  auto const rendered_prefix = render_text(prefix, policy);
  if (rendered.empty()) return prefix_first ? rendered_prefix : std::string{};

  std::string out;
  out.reserve(rendered.size() + rendered_prefix.size());

  std::size_t start = 0;
  bool first_line = true;
  while (start < rendered.size())
  {
    if (prefix_first || !first_line)
    {
      out += rendered_prefix;
    }

    auto const newline = rendered.find('\n', start);
    if (newline == std::string::npos)
    {
      out += rendered.substr(start);
      break;
    }

    out += rendered.substr(start, newline - start + 1);
    start = newline + 1;
    first_line = false;
  }

  return out;
}

std::string indent_text(std::string_view text, std::size_t spaces, output_policy const& policy, bool indent_first)
{
  return prefix_lines(text, std::string(spaces, ' '), policy, indent_first);
}

std::vector<std::string> wrap_text(std::string_view text, std::size_t max_width, output_policy const& policy)
{
  auto const rendered = render_text(text, policy);
  if (max_width == 0) return {std::string(rendered)};

  auto const is_breakable_space = [](decoded_codepoint const& decoded) {
    return decoded.valid && (decoded.value == U' ' || decoded.value == U'\t');
  };

  auto refresh_last_break = [&](std::string_view line, std::size_t& break_begin, std::size_t& break_end) {
    break_begin = std::string::npos;
    break_end = std::string::npos;
    for (std::size_t offset = 0; offset < line.size();)
    {
      auto const decoded = decode_next(line, offset);
      if (is_breakable_space(decoded))
      {
        break_begin = offset;
        break_end = offset + decoded.bytes;
      }
      offset += decoded.bytes;
    }
  };

  auto const trim_leading_spaces = [&](std::string value) {
    std::size_t offset = 0;
    while (offset < value.size())
    {
      auto const decoded = decode_next(value, offset);
      if (!is_breakable_space(decoded)) break;
      offset += decoded.bytes;
    }
    value.erase(0, offset);
    return value;
  };

  std::vector<std::string> lines;
  std::string current;
  std::size_t used = 0;
  std::size_t column = 0;
  std::size_t break_begin = std::string::npos;
  std::size_t break_end = std::string::npos;

  for (std::size_t offset = 0; offset < rendered.size();)
  {
    std::size_t ansi_length = 0;
    if (ansi_sequence_at(rendered, offset, ansi_length))
    {
      current.append(rendered.substr(offset, ansi_length));
      offset += ansi_length;
      continue;
    }

    auto const decoded = decode_next(rendered, offset);
    if (decoded.valid && decoded.value == U'\n')
    {
      lines.push_back(current);
      current.clear();
      used = 0;
      column = 0;
      break_begin = std::string::npos;
      break_end = std::string::npos;
      offset += decoded.bytes;
      continue;
    }

    std::size_t unit_width = decoded.bytes;
    if (policy.width == width_mode::display_cells)
    {
      if (!decoded.valid)
        unit_width = 1;
      else if (decoded.value == U'\t')
        unit_width = tab_advance(column, policy.tab_width);
      else
      {
        unit_width = codepoint_width_at(rendered, decoded, offset + decoded.bytes, policy);
      }
    }

    if (used > 0 && used + unit_width > max_width)
    {
      if (is_breakable_space(decoded))
      {
        lines.push_back(current);
        current.clear();
        used = 0;
        column = 0;
        break_begin = std::string::npos;
        break_end = std::string::npos;
        offset += decoded.bytes;
        continue;
      }
      if (break_begin != std::string::npos)
      {
        lines.push_back(current.substr(0, break_begin));
        current = trim_leading_spaces(current.substr(break_end));
        used = width_of_rendered(current, policy);
        column = used;
        refresh_last_break(current, break_begin, break_end);
      }
      else
      {
        lines.push_back(current);
        current.clear();
        used = 0;
        column = 0;
        break_begin = std::string::npos;
        break_end = std::string::npos;
      }
      if (current.empty() && is_breakable_space(decoded))
      {
        offset += decoded.bytes;
        continue;
      }
    }

    auto const current_offset = current.size();
    current.append(rendered.substr(offset, decoded.bytes));
    used += unit_width;
    column += unit_width;
    if (is_breakable_space(decoded))
    {
      break_begin = current_offset;
      break_end = current.size();
    }
    offset += decoded.bytes;
  }

  lines.push_back(current);
  return lines;
}

} // namespace uni20::presentation
