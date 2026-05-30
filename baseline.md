# Garbage Classification Class

## Goal - Get this notebook running on Nicla Vision

- Use GarbageClassification_CNN.ipynb as the source material to begin

- Analyze step 3-4.

- Generate files for all the possible TFLite QUantization options.

  - If plain int8 is failing, the most likely culprits are ops that lack int8 kernels. The "full int8 with float I/O" version (option 4) is more permissive and often works when strict int8 doesn't. Float16 is another solid option if you're not targeting microcontrollers.

- Write output to assignment_4_baseline.ipynb

#---


