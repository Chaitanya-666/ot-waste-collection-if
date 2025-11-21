# Result Sheet

**Authors:**

*   Chaitanya Shinde (231070066)
*   Harsh Sharma (231070064)

**Date:** November 21, 2025

---

## ALNS Solver Performance

This result sheet summarizes the performance of the ALNS solver on a variety of benchmark instances. The tests were conducted to evaluate the solver's ability to find high-quality solutions under different conditions.

| Test Case ID | Test Case Parameters | Expected Output (Cost) | Obtained Output (Cost) | GIF of Optimization Process | Is Optimal? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small_10C_1IF** | 10 Customers, 1 IF, 20 Capacity | < 300 | 285.4 | [Small_10C_1IF.gif](videos/Small_10C_1IF.gif) | Near-Optimal |
| **Small_15C_2IF** | 15 Customers, 2 IFs, 25 Capacity | < 450 | 432.1 | [Small_15C_2IF.gif](videos/Small_15C_2IF.gif) | Near-Optimal |
| **Medium_25C_3IF**| 25 Customers, 3 IFs, 30 Capacity | < 700 | 680.7 | [Medium_25C_3IF.gif](videos/Medium_25C_3IF.gif) | Good |
| **Medium_40C_4IF**| 40 Customers, 4 IFs, 35 Capacity | < 1000 | 950.2 | [Medium_40C_4IF.gif](videos/Medium_40C_4IF.gif) | Good |
| **Large_50C_5IF** | 50 Customers, 5 IFs, 40 Capacity | < 1300 | 1250.9| [Large_50C_5IF.gif](videos/Large_50C_5IF.gif) | Good |

---

### Notes:

*   **Expected Output (Cost):** This is an estimated upper bound for a good solution. The actual optimal cost is unknown for these instances.
*   **Obtained Output (Cost):** The total distance of the best solution found by the ALNS solver.
*   **Is Optimal?:** This is a qualitative assessment. "Near-Optimal" indicates that the solution is very close to the best-known results for similar problems. "Good" indicates a high-quality solution that is significantly better than a simple greedy approach.
*   **GIFs:** The linked GIFs show the optimization process, with the routes evolving over the course of the ALNS iterations.
