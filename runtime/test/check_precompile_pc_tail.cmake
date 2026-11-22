# gta6rt --precompile-pipelines with a --pc-tail-ubo compute shader (128-byte push constants + set-1 tail) alongside a
# 144-byte one: on 256-byte devices both compile and journal; on 128-byte devices (maxPushConstantsSize == 128) the
# --pc-tail-ubo entry still builds and journals while only the 144-byte entry is left out as unbuildable.
#   cmake -DGTA6RT=<gta6rt.exe> [-DRUNNER=<wine>] -DSPV_TAIL=<tail .spv> -DSPV_FULL=<full .spv>
#         -DTEST_DIR=<scratch folder> -P check_precompile_pc_tail.cmake
cmake_minimum_required(VERSION 3.24)

foreach(var GTA6RT SPV_TAIL SPV_FULL TEST_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "check_precompile_pc_tail.cmake: -D${var}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/shaders")
file(COPY_FILE "${SPV_TAIL}" "${TEST_DIR}/shaders/precompile_tail.spv")
file(COPY_FILE "${SPV_FULL}" "${TEST_DIR}/shaders/precompile_full.spv")
string(REPEAT "c" 64 hash_tail)
string(REPEAT "d" 64 hash_full)
set(tail_meta "\"stage\":\"cs\",\"bindings\":[],\"push_constant_bytes\":128,\"pc_tail\":{\"set\":1,\"binding\":0,\"bytes\":16},\"local_size\":[64,1,1],\"wave64\":false")
set(full_meta "\"stage\":\"cs\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[64,1,1],\"wave64\":false")
file(WRITE "${TEST_DIR}/shaders/index.json"
     "{\"precompile_tail\":{\"fssm_sha256\":\"${hash_tail}\",${tail_meta}},"
     "\"precompile_full\":{\"fssm_sha256\":\"${hash_full}\",${full_meta}}}\n")

foreach(var LEONIDA_PIPELINE_JOURNAL LEONIDA_PIPELINE_CACHE_BYTES LEONIDA_PIPELINE_JOURNAL_BYTES
            LEONIDA_PIPELINE_JOURNAL_SEED_DIR LEONIDA_COMPILE_HELPER LEONIDA_LAZY_COMPUTE LEONIDA_SKIP_COMPUTE)
  unset(ENV{${var}})
endforeach()
file(TO_NATIVE_PATH "${TEST_DIR}/shaders" shaders)
set(ENV{LEONIDA_SHADERS} "${shaders}")

set(failures "")
set(all_output "")
function(precompile label cache)
  file(TO_NATIVE_PATH "${TEST_DIR}/${cache}" cache_dir)
  set(ENV{LEONIDA_PIPELINE_CACHE_DIR} "${cache_dir}")
  execute_process(COMMAND ${RUNNER} "${GTA6RT}" --precompile-pipelines --threads 1 ${ARGN}
                  WORKING_DIRECTORY "${TEST_DIR}"
                  TIMEOUT 90
                  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
  string(REPLACE "\r\n" "\n" output "\n${output}\n")
  set(out "${output}" PARENT_SCOPE)
  set(code "${result}" PARENT_SCOPE)
  set(all_output "${all_output}--- ${label} (exit ${result}) ---${output}" PARENT_SCOPE)
endfunction()
function(expect label pattern)
  if(NOT out MATCHES "${pattern}")
    set(failures "${failures}${label}: no match for '${pattern}'\n" PARENT_SCOPE)
  endif()
endfunction()
function(expect_not label pattern)
  if(out MATCHES "${pattern}")
    set(failures "${failures}${label}: unexpected match for '${pattern}'\n" PARENT_SCOPE)
  endif()
endfunction()
function(expect_exit label want)
  if(NOT "${code}" STREQUAL "${want}")
    set(failures "${failures}${label}: exit ${code}, not ${want}\n" PARENT_SCOPE)
  endif()
endfunction()
function(journal_files cache var)
  file(GLOB files "${TEST_DIR}/${cache}/pipeline_cache.d/cs-*.bin")
  list(LENGTH files count)
  set(${var} ${count} PARENT_SCOPE)
endfunction()

precompile("first" cache)
if(code STREQUAL "1" AND out MATCHES "the Vulkan device could not be created")
  message("SKIP: no Vulkan device for the precompile here${all_output}")
  return()
endif()
expect_exit("first" 0)
expect_not("first" "Validation Error|VUID-")

if(out MATCHES "needs 144 push-constant bytes")
  set(want_built 1)
  set(want_unbuildable 1)
  expect("first" "\n\\[precompile\\] precompile_full: not built \\(needs 144 push-constant bytes\\)\n")
else()
  set(want_built 2)
  set(want_unbuildable 0)
endif()

expect("first" "\n\\[precompile\\] ${want_built} compute entries to build \\([^)]*, 0 left out by --skip-listed, ${want_unbuildable} not buildable\\)")
expect("first" "\n\\[precompile\\] done: ${want_built} built \\([0-9]+ compiled, [0-9]+ from a cache\\), 0 journaled already, 0 being built by another process, 0 failed; left out: 0 of 0 listed in LEONIDA_SKIP_COMPUTE, ${want_unbuildable} not buildable, 0 filtered out; in [0-9]+ s\n")

set(journaled TRUE)
if(out MATCHES "compiled \\(not journaled\\)")
  set(journaled FALSE)
  message("note: the driver gave no cache data, so the journal-file checks are left out")
else()
  journal_files(cache count)
  if(NOT count EQUAL ${want_built})
    string(APPEND failures "first: ${count} journal files, not ${want_built}\n")
  endif()
endif()

if(journaled)
  precompile("again" cache)
  expect_exit("again" 0)
  expect_not("again" "Validation Error|VUID-")
  expect("again" "\n\\[precompile\\] done: 0 built \\(0 compiled, 0 from a cache\\), ${want_built} journaled already, 0 being built by another process, 0 failed; left out: 0 of 0 listed in LEONIDA_SKIP_COMPUTE, ${want_unbuildable} not buildable, 0 filtered out; in [0-9]+ s\n")
endif()

if(failures)
  message(FATAL_ERROR "gta6rt --precompile-pipelines --pc-tail-ubo:\n${failures}${all_output}")
endif()
message("precompile --pc-tail-ubo: built ${want_built}, ${want_unbuildable} not buildable${all_output}")
