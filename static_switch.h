#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                     \
  [&] {                                                                        \
    if (COND) {                                                                \
      constexpr static bool CONST_NAME = true;                                 \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      constexpr static bool CONST_NAME = false;                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()

#define VALUE_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                             \
  [&] {                                                                        \
    if (NUM_SPLITS == 96) {                                                    \
      constexpr static int NAME = 96;                                          \
      return __VA_ARGS__();                                                    \
    } else if (NUM_SPLITS == 128) {                                            \
      constexpr static int NAME = 128;                                         \
      return __VA_ARGS__();                                                    \
    }                                                                          \
  }()
