# Ford Line Health Monitor: results

Built for Ford Motor Company. Prepared by K Mohan Kumar.

## Key figures

- **Machines at risk:** 3 (of 24 on the line)
- **Highest 7-day risk:** 99% (STAM-03, Stamping press)
- **Downtime avoided:** 51 hrs (Unplanned stops turned into planned work)
- **Saving:** $6.1M (At $120K per downtime hour)

## Algorithm, as it ran

1. **Collect sensor and maintenance data**: 24 machines x 120 days, 2,600 historical windows
2. **Build health features**: 3 sensor z-scores per machine-day vs its first 30 days
3. **Anomaly detection**: 3 machines with health index above 3
4. **Remaining useful life**: Shortest remaining life: STAM-03 at 0 days
5. **7-day failure probability**: Model AUC 0.901; top risk STAM-03 at 99%
6. **Failure patterns**: Bearing and alignment wear 46%, Overheating 29%, Electrical and motor faults 24%
7. **Schedule maintenance**: 3 jobs booked, 1 needing an immediate stop
8. **Work orders**: Ready

## Recommended actions

1. **STAM-03: bearing or alignment wear.** Stamping press at 99% risk of failing within 7 days. No planned window comes early enough; stop at the next shift change. *Impact: Stop now*
2. **CONV-05: motor or electrical fault.** Conveyor drive at 38% risk of failing within 7 days. Book the 3-hour job into day 8, midweek gap. *Impact: Day 8, Midweek gap*
3. **WELD-06: cooling or lubrication fault.** Welding robot at 22% risk of failing within 7 days. Book the 4-hour job into day 19, sunday shutdown. *Impact: Day 19, Sunday shutdown*
4. **Target bearing and alignment wear across the line.** 46% of past failures follow this pattern. Add checks for it to every Sunday shutdown. *Impact: Prevent repeats*

_Generated from sample data. Figures are illustrative._
