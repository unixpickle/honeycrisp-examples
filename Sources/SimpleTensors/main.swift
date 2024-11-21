import Honeycrisp

@main
struct Main {
  static func main() async {
    do {

      // Create a 2x3 matrix:
      //   1  2  3
      //   4  5  6
      let matrix = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])

      // Add 1 to every element, and get the new data and shape
      let matrixPlus1 = matrix + 1
      print("plus 1 shape:", matrixPlus1.shape, "data:", try await matrixPlus1.floats())

      let sumOfColumns = matrix.sum(axis: 1)
      print("sum of columns shape:", sumOfColumns.shape, "data:", try await sumOfColumns.floats())

      let xFloat = Tensor(data: [1, 2, 3], shape: [3], dtype: .float32)
      let xInt = xFloat.cast(.int64)

      // Will fail with an error, due to mismatched data types:
      // let _ = xFloat + xInt
      print("floats:", try await xFloat.floats())
      print("ints:", try await xInt.ints())

    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
