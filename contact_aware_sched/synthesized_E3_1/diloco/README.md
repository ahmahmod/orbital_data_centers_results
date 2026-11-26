# DiLoCo — E3_1

Use `METHOD=diloco8 CASE=off ./run.sh` and `METHOD=diloco8 CASE=aware ./run.sh`; repeat for `diloco64`. Outer learning rate 0.3 and outer momentum 0.9 are preserved. This contribution defers outer synchronization without changing H or doing extra local optimizer steps during waiting.
