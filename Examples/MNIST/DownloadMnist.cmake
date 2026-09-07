if(NOT DEFINED MNIST_DATA_DIRECTORY)
    message(FATAL_ERROR "MNIST_DATA_DIRECTORY was not provided")
endif()

file(MAKE_DIRECTORY "${MNIST_DATA_DIRECTORY}")

set(MNIST_BASE_URL "https://storage.googleapis.com/cvdf-datasets/mnist")
set(MNIST_FILES
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
)

foreach(MNIST_FILE IN LISTS MNIST_FILES)
    set(MNIST_OUTPUT "${MNIST_DATA_DIRECTORY}/${MNIST_FILE}")
    if(EXISTS "${MNIST_OUTPUT}")
        continue()
    endif()

    set(MNIST_ARCHIVE "${MNIST_OUTPUT}.gz")
    if(NOT EXISTS "${MNIST_ARCHIVE}")
        message(STATUS "Downloading ${MNIST_FILE}.gz")
        file(
            DOWNLOAD
            "${MNIST_BASE_URL}/${MNIST_FILE}.gz"
            "${MNIST_ARCHIVE}"
            STATUS MNIST_DOWNLOAD_STATUS
            SHOW_PROGRESS
            TLS_VERIFY ON
        )

        list(GET MNIST_DOWNLOAD_STATUS 0 MNIST_DOWNLOAD_RESULT)
        list(GET MNIST_DOWNLOAD_STATUS 1 MNIST_DOWNLOAD_MESSAGE)
        if(NOT MNIST_DOWNLOAD_RESULT EQUAL 0)
            file(REMOVE "${MNIST_ARCHIVE}")
            message(FATAL_ERROR
                "Failed to download ${MNIST_FILE}.gz: ${MNIST_DOWNLOAD_MESSAGE}"
            )
        endif()
    endif()

    message(STATUS "Extracting ${MNIST_FILE}.gz")
    file(
        ARCHIVE_EXTRACT
        INPUT "${MNIST_ARCHIVE}"
        DESTINATION "${MNIST_DATA_DIRECTORY}"
    )

    if(NOT EXISTS "${MNIST_OUTPUT}")
        message(FATAL_ERROR "Extraction did not produce ${MNIST_OUTPUT}")
    endif()
endforeach()
