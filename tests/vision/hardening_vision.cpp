// tests/vision/hardening_vision.cpp - a picture of any shape lands inside the token budget, and a side past the cap
// is refused at the header.
//
// The pixel budget (preprocess_limit_test.cpp) bounds w * h; these cases cover what it leaves: a long thin picture,
// whose short side the canvas rounds up to one merge unit, and a geometry with a small or odd token interval.
#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::vision;

namespace {

int failures = 0;

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);                            \
            ++failures;                                                                    \
        }                                                                                  \
    } while (0)

int64_t tokens(const Size& s, const Geometry& g) {
    const int f = g.patch * g.merge;
    return (int64_t) (s.w / f) * (s.h / f);
}

// the canvas is a positive multiple of patch * merge and holds between min_tokens and max_tokens merged tokens
bool in_budget(int w, int h, const Geometry& g) {
    const Size s = fit_size(w, h, g);
    const int f = g.patch * g.merge;
    const int64_t t = tokens(s, g);
    const bool ok = s.w > 0 && s.h > 0 && s.w % f == 0 && s.h % f == 0 && t >= g.min_tokens && t <= g.max_tokens;
    if (!ok)
        std::printf("  %dx%d with patch %d merge %d tokens %d..%d gave %dx%d (%lld tokens)\n", w, h, g.patch, g.merge,
                    g.min_tokens, g.max_tokens, s.w, s.h, (long long) t);
    return ok;
}

// a 24-bit BMP of w x 1 pixels, so a decode of an extreme width is real, not a header with no data
std::vector<uint8_t> bmp_row(int w, int h) {
    const uint32_t stride = ((uint32_t) w * 3 + 3) & ~3u, size = 54 + stride * (uint32_t) h;
    std::vector<uint8_t> b(size, 0x80);
    auto le32 = [&](size_t at, uint32_t v) { for (int i = 0; i < 4; ++i) b[at + i] = (uint8_t) (v >> (8 * i)); };
    b[0] = 'B';
    b[1] = 'M';
    le32(2, size);
    le32(6, 0);
    le32(10, 54);
    le32(14, 40);
    le32(18, (uint32_t) w);
    le32(22, (uint32_t) h);
    b[26] = 1;
    b[27] = 0;
    b[28] = 24;
    b[29] = 0;
    le32(30, 0);
    le32(34, stride * (uint32_t) h);
    for (size_t i = 38; i < 54; ++i) b[i] = 0;
    return b;
}

}  // namespace

int main() {
    const Geometry defaults;

    // every shape from a dot to the longest strip, at the geometry the encoder uses and at odd ones
    const int sides[] = {1, 2, 3, 7, 16, 31, 32, 33, 100, 169, 640, 1023, 1920, 4096, 8191, 32768, 131073, 1 << 24,
                         1 << 26};
    struct Geo { int patch, merge, min_tokens, max_tokens; };
    const Geo geos[] = {{16, 2, 8, 4096}, {16, 2, 8, 8}, {16, 2, 1, 1}, {16, 2, 5, 7}, {14, 2, 4, 1280},
                        {16, 1, 8, 4096}, {8, 3, 2, 64}, {16, 2, 8, 65536}};
    for (const Geo& gg : geos) {
        Geometry g;
        g.patch = gg.patch;
        g.merge = gg.merge;
        g.min_tokens = gg.min_tokens;
        g.max_tokens = gg.max_tokens;
        for (int w : sides)
            for (int h : sides) CHECK(in_budget(w, h, g));
    }
    // a dense sweep of the small sizes, where the rounding steps and both branches meet
    for (int w = 1; w <= 700; ++w)
        for (int h = 1; h <= 700; h += (h < 100 ? 1 : 13)) CHECK(in_budget(w, h, defaults));

    // a picture 1 pixel wide and 64 Mi tall passes the pixel budget; its canvas must still be inside the token budget
    CHECK(fit_size(1, 1 << 26, defaults) == (Size{32, 131072}));
    CHECK(fit_size(1 << 26, 1, defaults) == (Size{131072, 32}));
    // the two shapes a small interval forces onto one row
    {
        Geometry g;
        g.min_tokens = g.max_tokens = 8;
        CHECK(fit_size(32768, 1, g) == (Size{256, 32}));
        CHECK(fit_size(1, 32768, g) == (Size{32, 256}));
    }

    // sizes the previous float arithmetic put one 32-pixel step off an exact boundary (the real ratio is 64/13 here)
    CHECK(fit_size(2, 169, defaults) == (Size{32, 832}));
    CHECK(fit_size(10, 45, defaults) == (Size{64, 192}));
    // ordinary pictures: the sizes the encoder has always been given
    CHECK(fit_size(640, 480, defaults) == (Size{640, 480}));
    CHECK(fit_size(1920, 1080, defaults) == (Size{1920, 1088}));
    CHECK(fit_size(4000, 3000, defaults) == (Size{2336, 1760}));
    CHECK(fit_size(100, 100, defaults) == (Size{96, 96}));
    CHECK(fit_size(1, 1, defaults) == (Size{96, 96}));

    // no pixels, or a geometry with no size to give
    CHECK(fit_size(0, 5, defaults) == (Size{0, 0}));
    CHECK(fit_size(5, -1, defaults) == (Size{0, 0}));
    {
        Geometry g;
        g.patch = 0;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
        g = Geometry{};
        g.merge = -2;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
        g = Geometry{};
        g.min_tokens = 0;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
        g = Geometry{};
        g.min_tokens = 10;
        g.max_tokens = 9;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
        g = Geometry{};
        g.patch = INT32_MAX;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
        g = Geometry{};
        g.max_tokens = INT32_MAX;
        CHECK(fit_size(5, 5, g) == (Size{0, 0}));
    }

    // the side cap, at the header and on a real decode
    {
        Image img;
        std::string err;
        const auto wide = bmp_row(kMaxImageSide + 1, 1);
        CHECK(!decode_image(wide.data(), wide.size(), img, err));
        CHECK(err.find("a side may not pass") != std::string::npos);
        const auto tall = bmp_row(1, kMaxImageSide + 1);
        err.clear();
        CHECK(!decode_image(tall.data(), tall.size(), img, err));
        CHECK(err.find("a side may not pass") != std::string::npos);

        const auto strip = bmp_row(kMaxImageSide, 1);
        err.clear();
        CHECK(decode_image(strip.data(), strip.size(), img, err));
        CHECK(img.w == kMaxImageSide && img.h == 1);
        const Size sz = fit_size(img.w, img.h, defaults);
        CHECK(tokens(sz, defaults) >= defaults.min_tokens && tokens(sz, defaults) <= defaults.max_tokens);
        const Image canvas = to_canvas(img, sz);
        CHECK(canvas.w == sz.w && canvas.h == sz.h && canvas.rgb.size() == (size_t) sz.w * sz.h * 3);
    }

    if (failures != 0) return 1;
    std::printf("vision hardening: ok\n");
    return 0;
}
