# asm/CMakeLists.txt

# pick up the same compiler and flags as the main project
set(CXX           ${CMAKE_CXX_COMPILER})
set(CXXFLAGS      ${CMAKE_CXX_FLAGS})
set(CXXFLAGS_DBG  ${CMAKE_CXX_FLAGS_DEBUG})
set(CXXFLAGS_REL  ${CMAKE_CXX_FLAGS_RELEASE})
set(STD_OPTION    ${CMAKE_CXX_STANDARD_COMPILE_OPTION})
set(EXT_OPTION    ${CMAKE_CXX_EXTENSIONS_COMPILE_OPTION})

# glob .cpp snippets
file(GLOB ASM_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

# build a .s for each .cpp
set(ASM_OUTS)
foreach(src IN LISTS ASM_SRCS)
  get_filename_component(base ${src} NAME_WE)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/${base}.s)
  list(APPEND ASM_OUTS ${out})

  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CXX}
            ${CXXFLAGS}
            $<$<CONFIG:DEBUG>:${CXXFLAGS_DBG}>
            $<$<CONFIG:RELEASE>:${CXXFLAGS_REL}>
            -S
            ${STD_OPTION}
            ${EXT_OPTION}
            -o ${out}
            ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
    COMMENT "→ Assembling ${src}"
    VERBATIM
  )
endforeach()

add_custom_target(asm
  DEPENDS ${ASM_OUTS}
  COMMENT "Build all assembly snippets"
)
