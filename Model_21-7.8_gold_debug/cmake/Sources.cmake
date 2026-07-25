# Centralized source list for the icar project.
# Keep this file focused on project files only.

set(ICAR_CORE_SOURCES
    res/include/video_get.cpp
    src/common/utils.cpp
    src/inference/ocr/ocr_api.cpp
    src/inference/ocr/ocr_detector.cpp
    src/inference/ppseg/ppseg_detector.cpp
    src/inference/ppyoloe/ppyoloe_detector.cpp
    src/inference/ppyoloe/ppyoloe_postprocess.cpp
    src/imgprocess/imgProcess.cpp
    src/imgprocess/LineTracker.cpp
    src/imgprocess/TrajectoryFitter.cpp
    src/imgprocess/transform.cpp
    src/special/branch.cpp
    src/special/go_stop.cpp
    src/special/human.cpp
    src/special/light.cpp
    src/special/objects.cpp
    src/standard/general.cpp
    src/standard/standard.cpp
    src/standard/fuzzy.cpp
    src/thread/ocr_thread.cpp
    src/thread/thread.cpp
)

set(ICAR_APP_SOURCES
    main.cpp
    ${ICAR_CORE_SOURCES}
)


