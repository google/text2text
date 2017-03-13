package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//textsum/...",
    ],
)

py_library(
    name = "model",
    srcs = glob(["model/*.py"]),
    deps = [
        ":library",
    ],
)

py_library(
    name = "library",
    srcs = ["library.py"],
)

py_binary(
    name = "main",
    srcs = [
        "main.py",
    ],
    deps = [
        ":config",
        ":data",
        ":decode",
    ],
)

py_library(
    name = "batch_reader",
    srcs = glob(["batch_reader/*.py"]),
    deps = [
        ":data",
    ],
)

py_library(
    name = "beam_search",
    srcs = ["beam_search.py"],
)

py_library(
    name = "decode",
    srcs = ["decode.py"],
    deps = [
        ":beam_search",
        ":data",
        ":metrics",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
)

py_library(
    name = "config",
    srcs = glob(["config/*.py"]),
    deps = [
        ":batch_reader",
        ":model",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
)
