import Honeycrisp

@main
struct Main {
  static func main() async {
    do {
      Backend.defaultBackend = try MPSBackend()  // Use the GPU by default
      let cpuBackend = CPUBackend()
      let x = Tensor(rand: [128, 128])  // Performed on GPU
      let y = cpuBackend.use { x + 3 }  // Performed on CPU
      let z = y - 3  // Performed on GPU
      print("input mean:", try await x.mean().item())
      print("output mean (should match):", try await z.mean().item())
    } catch {
      print("FATAL ERROR: \(error)")
    }
  }
}
