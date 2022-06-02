if(OPENVINO_ROOT)
  set(InferenceEngine_DIR ${OPENVINO_ROOT}/cmake)
  set(ngraph_DIR ${OPENVINO_ROOT}/cmake)
elseif($ENV{OPENVINO_ROOT})
  set(InferenceEngine_DIR $ENV{OPENVINO_ROOT}/cmake)
  set(ngraph_DIR $ENV{OPENVINO_ROOT}/cmake)
endif()

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

list(APPEND XRT_OPENVINO_THIRD_PARTY_LIBRARIES IE::inference_engine
     ${NGRAPH_LIBRARIES})
