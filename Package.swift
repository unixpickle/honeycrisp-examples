// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "HoneycrispExamples",
  platforms: [
    .macOS(.v13)
  ],
  products: [
    .library(
      name: "MNIST",
      targets: ["MNIST"])
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-crypto.git", "1.0.0"..<"4.0.0"),
    .package(url: "https://github.com/1024jp/GzipSwift", "6.0.0"..<"6.1.0"),
    .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.9.0"),
    .package(url: "https://github.com/unixpickle/honeycrisp.git", from: "0.0.7"),
  ],
  targets: [
    .executableTarget(
      name: "SimpleTensors",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]
    ),
    .executableTarget(
      name: "Backends",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]
    ),
    .executableTarget(
      name: "Trainable",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp")
      ]
    ),
    .target(
      name: "MNIST",
      dependencies: [
        .product(name: "AsyncHTTPClient", package: "async-http-client"),
        .product(name: "Gzip", package: "GzipSwift"),
        .product(name: "Crypto", package: "swift-crypto"),
      ]),
    .executableTarget(
      name: "MNISTExample",
      dependencies: [
        "MNIST",
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]),
    .executableTarget(
      name: "MNISTGenExample",
      dependencies: [
        "MNIST",
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "HCBacktrace", package: "honeycrisp"),
      ]),
  ]
)
