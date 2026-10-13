cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# esp_wifi_remote selects its generated Kconfig using ESP_IDF_VERSION, while
# ESP-IDF exports only IDF_VERSION. Supply the major.minor branch it expects.
if(NOT DEFINED ENV{ESP_IDF_VERSION})
    set(ENV{ESP_IDF_VERSION} "${IDF_VERSION_MAJOR}.${IDF_VERSION_MINOR}")
endif()

project(h264_stream)
