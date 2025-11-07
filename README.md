# 🧠 Region-Based Register Allocator

A Python-based compiler optimization project that implements **region-based register allocation** using **control flow analysis**, **liveness detection**, and **graph coloring**.  
It visualizes allocation decisions and register pressure to demonstrate how modern compilers manage CPU registers efficiently.

---

## 📘 Overview

This project simulates how a compiler backend performs **register allocation** — assigning program variables to a limited number of CPU registers to minimize memory access (spilling).  
It introduces **region partitioning** (using strongly connected components) to optimize allocation in complex programs.

**Key Features**
- Builds **Control Flow Graph (CFG)** from Intermediate Representation (IR)
- Performs **Live Variable Analysis**
- Divides code into **Regions** for localized optimization
- Constructs **Interference Graphs** for register conflict detection
- Uses **Graph Coloring** for register assignment
- Visualizes **Register Pressure** and allocation results with Matplotlib

---

## 🧩 System Architecture


IR Input → Parsing → CFG Construction → Liveness Analysis → Region Partitioning →
Graph Coloring Register Allocation → Code Generation → Visualization


---

## ⚙️ Requirements

- Python 3.8 or above  
- Install dependencies:
  ```bash
  pip install networkx matplotlib


📂 Project Structure

RegionBasedAllocator/
│
├── region_alloc_cfg.py        # Main Python script
├── program.ir                 # Sample IR input file
├── outputs/                   # Generated graphs and plots
│   ├── region_0_graph.png
│   ├── region_1_graph.png
│   └── register_pressure.png
└── README.md                  # Project documentation


🚀 Steps to Execute

Step 1: Prepare Environment

Ensure Python 3.8+ is installed, then install dependencies:

pip install networkx matplotlib

Step 2: Create Input IR File

Create a file named program.ir and add sample IR code:

start:
i = 0
sum = 0
L1:
temp = i * 1
if temp goto L2
goto L3
L2:
sum = sum + i
i = i + 1
goto L1
L3:
print sum

Step 3: Run the Allocator

Run the script:

python region_alloc_cfg.py program.ir

Step 4: Observe Output

<img width="600" height="450" alt="region_0_graph" src="https://github.com/user-attachments/assets/19243d34-8285-486f-89f0-ccfa3784a6ad" />

<img width="600" height="450" alt="region_1_graph" src="https://github.com/user-attachments/assets/f755b9d6-ac76-40fa-961a-327c960a69b1" />

<img width="600" height="450" alt="region_3_graph" src="https://github.com/user-attachments/assets/a7714d96-7fe7-4da9-8c56-eaa431b9aa64" />

Console displays:

CFG construction

Liveness sets

Region partitions

Register allocations

Optimized code

Generated visualization files:

region_*.png — interference graphs

register_pressure.png — register pressure over time

<img width="900" height="450" alt="register_pressure" src="https://github.com/user-attachments/assets/b20abeca-3de2-48ee-ab68-2b4fe03114db" />


🧠 Example Output

Blocks formed: 5
Regions (SCCs): [[0], [1, 3], [2], [4]]

=== Allocation Summary ===
Region 1 blocks [1, 3]:
  i   -> r1
  sum -> r2
  temp-> r3

Globally spilled vars: []

📊 Visualization

Example outputs generated:

region_0_graph.png – shows interference between variables

register_pressure.png – displays live variable count across code regions



🔍 Validation

Confirms correct region detection and live variable tracking

Ensures registers are efficiently assigned with minimal spilling

Visual outputs confirm correctness of region-based partitioning


🧱 Future Enhancements

Implement spill cost heuristics

Support SSA (Static Single Assignment) form

Integrate loop-based region optimization

Extend for real LLVM IR or assembly backends
