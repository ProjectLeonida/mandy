# Names the Vulkan headers a configure builds against, in one STATUS line a GPU test record can quote:
#   -- Vulkan headers: <include dir> (VK_HEADER_VERSION <n>), <where they come from>
# The headers decide what the GPU test tools can ask the driver for, so two records of the same GPU and driver differ
# when their headers do: rebuilt against conda-forge's Vulkan 1.3 headers instead of the pin, run_gpu_tests.py on a
# Radeon 780M skipped ps_texoff, ps_texoff_forms and ps_texoff_implicit (225/248 passed instead of 228/248).
# Kept in its own file so tests/test_cmake_vulkan_headers_pin.py can run it with `cmake -P`.

# Sets dir_var to the first of include_dirs (the INTERFACE_INCLUDE_DIRECTORIES of Vulkan::Headers) that holds
# vulkan/vulkan_core.h or vulkan/vulkan.h, or to "" when none does, and version_var to its VK_HEADER_VERSION, or to
# "unknown". Generator expressions such as $<BUILD_INTERFACE:...> (FetchContent's target) are unwrapped.
function(leonida_vulkan_headers_find include_dirs dir_var version_var)
  set(dir "")
  set(header "")
  foreach(d IN LISTS include_dirs)
    string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" d "${d}")
    if(NOT d OR d MATCHES "^\\$<")
      continue()
    endif()
    foreach(name vulkan_core.h vulkan.h)
      if(EXISTS "${d}/vulkan/${name}")
        set(dir "${d}")
        set(header "${d}/vulkan/${name}")
        break()
      endif()
    endforeach()
    if(dir)
      break()
    endif()
  endforeach()
  set(version "unknown")
  if(header)
    file(STRINGS "${header}" line REGEX "^#define VK_HEADER_VERSION [0-9]+" LIMIT_COUNT 1)
    if(line MATCHES "VK_HEADER_VERSION ([0-9]+)")
      set(version "${CMAKE_MATCH_1}")
    endif()
  endif()
  set(${dir_var} "${dir}" PARENT_SCOPE)
  set(${version_var} "${version}" PARENT_SCOPE)
endfunction()

# include_dirs: the INTERFACE_INCLUDE_DIRECTORIES of Vulkan::Headers. origin: PIN (the hash-checked download of
# LEONIDA_VULKAN_HEADERS_PIN), PIN_CACHED (a copy already in FETCHCONTENT_BASE_DIR, which
# FETCHCONTENT_FULLY_DISCONNECTED=ON uses without downloading or checking an archive; leonida_vulkan_headers_check
# has confirmed only its version), SOURCE_DIR (FETCHCONTENT_SOURCE_DIR_VULKANHEADERS) or FOUND (anything else).
function(leonida_vulkan_headers_report include_dirs origin)
  leonida_vulkan_headers_find("${include_dirs}" dir version)
  if(NOT dir)
    set(dir "${include_dirs}")
  endif()
  # string(CONCAT) on quoted pieces keeps a ";" in the text, which a list would split on.
  if(origin STREQUAL "PIN")
    set(what "the pin: Khronos Vulkan-Headers ${LEONIDA_VULKAN_HEADERS_PIN}, hash-checked")
  elseif(origin STREQUAL "PIN_CACHED")
    string(CONCAT what "the pinned version, cached source: Khronos Vulkan-Headers ${LEONIDA_VULKAN_HEADERS_PIN}; "
                       "archive not checked in this configure (FETCHCONTENT_FULLY_DISCONNECTED=ON)")
  elseif(origin STREQUAL "SOURCE_DIR")
    string(CONCAT what "from FETCHCONTENT_SOURCE_DIR_VULKANHEADERS: a local copy used in place of the pin, not "
                       "checked against it")
  else()
    string(CONCAT what "found on this machine, not the pin (-DLEONIDA_VULKAN_HEADERS_FETCH=ON builds against the "
                       "pinned Khronos Vulkan-Headers)")
  endif()
  message(STATUS "Vulkan headers: ${dir} (VK_HEADER_VERSION ${version}), ${what}")
endfunction()

# Fails the configure unless Vulkan::Headers (include_dirs) is the Vulkan-Headers copy FetchContent populated in
# source_dir (empty when it populated none). FetchContent_MakeAvailable(VulkanHeaders) can return other headers
# without populating anything: a dependency provider can answer it, and so can find_package() when an earlier
# FetchContent_Declare(VulkanHeaders) asked for that or when FETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS applies (an
# installed VulkanHeadersConfig.cmake with VK_HEADER_VERSION 111 was named as the pin that way). origin PIN or
# PIN_CACHED also needs the pinned version, LEONIDA_VULKAN_HEADERS_PIN_VERSION, which a build folder populated before
# the pin changed and configured with FETCHCONTENT_FULLY_DISCONNECTED=ON does not have.
function(leonida_vulkan_headers_check include_dirs source_dir origin)
  leonida_vulkan_headers_find("${include_dirs}" dir version)
  set(same FALSE)
  if(dir AND source_dir)
    file(REAL_PATH "${dir}" used)
    file(REAL_PATH "${source_dir}/include" wanted)
    if(used STREQUAL wanted)
      set(same TRUE)
    endif()
  endif()
  if(NOT same)
    if(source_dir)
      set(wanted "${source_dir}/include")
    else()
      set(wanted "a folder FetchContent populated (it populated none)")
    endif()
    message(FATAL_ERROR
      "-DLEONIDA_VULKAN_HEADERS_FETCH=ON or FETCHCONTENT_SOURCE_DIR_VULKANHEADERS asks for the Vulkan headers in "
      "${wanted}, but Vulkan::Headers includes \"${include_dirs}\". Something other than that download answered "
      "FetchContent_MakeAvailable(VulkanHeaders): a dependency provider (CMAKE_PROJECT_TOP_LEVEL_INCLUDES) or an "
      "earlier FetchContent_Declare(VulkanHeaders). Remove it, or configure without either of the two to build "
      "against the installed headers.")
  endif()
  if((origin STREQUAL "PIN" OR origin STREQUAL "PIN_CACHED") AND
     NOT version STREQUAL "${LEONIDA_VULKAN_HEADERS_PIN_VERSION}")
    message(FATAL_ERROR
      "${dir} has VK_HEADER_VERSION ${version}, but the pinned Khronos Vulkan-Headers ${LEONIDA_VULKAN_HEADERS_PIN} "
      "are version ${LEONIDA_VULKAN_HEADERS_PIN_VERSION}, so ${source_dir} is not the pin. A dependency provider may "
      "have populated it, or the build folder was populated before the pin changed and FETCHCONTENT_FULLY_DISCONNECTED "
      "is ON, which keeps the old copy: configure once with -DFETCHCONTENT_FULLY_DISCONNECTED=OFF to download the pin.")
  endif()
endfunction()
