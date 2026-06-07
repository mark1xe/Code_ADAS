cmake_minimum_required(VERSION 3.10)
project(lane_project)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

# Chuong trình 1
add_executable(run_tune run_tune.cpp)
# Chuong trình 2
add_executable(run_yolo run_yolo.cpp)
# Chuong trình 3
add_executable(run_tune_yolo run_tune_yolo.cpp)

target_link_libraries(run_tune
    ${OpenCV_LIBS}
    pthread
)
target_link_libraries(run_yolo
    ${OpenCV_LIBS}
    pthread
)
target_link_libraries(run_tune_yolo
    ${OpenCV_LIBS}
    pthread
    tensorflow-lite  # <--- Bắt buộc phải thêm dòng này để chạy TFLite
)