# Evaluation Modules Demonstration

In CoachAI+, we offer three **Evaluation Modules** for match data analysis, each focusing on different aspects of the game for in-depth insights and interpretation. Through these features, coaches and players can understand match details, identify potential strategies for improvement, and ultimately enhance performance.

Each Evaluation Module allows for match data import, the format would be like [demo.csv](...). You can directly use [demo.csv]() in our [web page](https://badminton.andyjjrt.cc)

![Website Preview](./Website%20Preview.png)

We will use the **Malaysia Masters 2020 Finals match between Kento Momota and Viktor Axelsen** as an example. In this match, Momota emerged victorious. By using our system, we will highlight Momota's strengths and potential weaknesses, demonstrating the effectiveness of the Evaluation Modules.

## Shot Evaluation

### Rally Length vs. Energy Expenditure Diagram
When we import the second set for analysis, the first visualization we see is the "Rally Length vs. Energy Expenditure Diagram." The line represents energy expenditure, and the bar chart represents the length of each rally.

![Rally Length vs. Energy Expenditure Diagram](./Energy%20Cost%20Per%20Rally%20And%20Ball%20Rounds%20Distribution.png)

We can clearly observe that Kento Momota tends to win when rallies are longer. Even when the rally count is lower, Momota's playing style tends to force Axelsen into greater energy expenditure.

### Shot Type Visualization
We can further analyze the shot placement distribution by shot type. In badminton, "smash" is usually the primary shot for scoring points. In the upper half of the chart, green represents winning shot placements, while red represents losing shot placements. In the lower half, green indicates player movements after winning shots, while red indicates player movements after losing shots:

![Shot Type Visualization](./Shot%20Type%20Visualization.png)

We observed that Momota's win rate is significantly higher on the left side when smashing, reflecting the fact that he is left-handed. However, it is noteworthy that Momota performs better when smashing straight.

### Rally Shot-by-Shot Momentum and Pace Plot
For each rally, we can use the system to closely observe how each shot contributed to Momota's points. Our **Evaluation Module** incorporates a deep learning model called [Shot Influence](https://github.com/wywyWang/CoachAI-Projects/tree/main/Shot%20Influence), which assesses shot-by-shot momentum. As shown in the left graph, it evaluates win probability changes for each shot. However, as a predictive model, it lacks interpretability. Therefore, we also include the **Pace Plot** (shown on the right). The pace value reflects active vs. passive play multiplied by pace—shots above the net are active, and those below are passive. This allows players to understand the reasoning behind win probability changes. The example below shows the 15th rally of set 2:

![Rally Shot-by-Shot Momentum and Pace Plot](./Rally%20Shot-by-Shot%20Momentum%20and%20Pace%20Plot.png)

We can see that Momota has a higher win rate when using net shots, and he seems to perform better when the pace is slower, which aligns with his style.

## Tactic Evaluation
In addition to observing shot types and rallies, coaches and players often focus on tactical usage. Definitions for tactics can be found in the [Badminton Tactic Definition](#Badminton-Tactic-Definition) section.

### Tactic Usage Bar Chart
In the bar chart, we can see that **Axelsen** relied heavily on "Full-Court Pressure", while **Momota** was more dependent on "Four-Corner". Full-Court Pressure rallies tend to be shorter than Four-Corner Play, which verifies that Axelsen preferred not to engage in long rallies against Momota.

![Tactic Usage Bar Chart](./Tactic%20Usage%20Bar%20Chart.png)
| Tactic       | abbreviation                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Full-Court Pressure | FCP|
Defensive Counterattack|DC|
Four-Corner| FC|
Forehand Lock| FhL|
Backhand Lock| BhL|
FrontCourt Lock|FcL|
BackCourt Lock| BcL|
Four-Corners Clear Drop| FCCD|
No tactic|No|



### Tactic Win Rate vs. Usage Scatter Plot
The Tactic Win Rate vs. Usage Scatter Plot further illustrates that Axelsen's Full-Court Pressure is his most formidable weapon. However, once the rally shifts to a Four-Corner, his win rate drops below 50%. This motivates him to adopt the Full-Court Pressure strategy early in short rallies to avoid extended exchanges. On the other hand, while Momota also excels in the Full-Court Pressure, his Defensive Counterattack also give him a win rate of over 75%, showcasing his expertise in turning defense into offense—a critical advantage against Axelsen.

![Tactic Win Rate vs. Usage Scatter Plot](./Tactic%20Win%20Rate%20vs.%20Usage%20Scatter%20Plot.png)


### Tactic Usage Pie Chart
The pie chart for Momota's tactic usage shows a clear reliance on Four-Corner, with over 50% of his plays using this strategy. This indicates that he is comfortable, and even thrives, in dragging his opponent into longer rallies.

![Tactic Usage Pie Chart](./Tactic%20Usage%20Pie%20Chart.png)

## Visualization & Winning / Losing Reason Statistics

In this module, we visualize the match progress by dividing the court into 10 areas (9 within the court and 1 out-of-bounds area). Using this segmentation, we analyze the distribution of winning and losing actions. Since importing data from only one set may not provide reliable statistical results, we imported all of Momota's past match data for analysis in this module:

![Visualization & Winning / Losing Reason Statistics](./Visualization%20&%20Winning-Losing%20Reason%20Statistics.png)

We can observe that Momota's winning shots tend to land on the left side, which aligns with the results from the Shot Evaluation. His smash is more effective when placed in a straight line. On the other hand, Momota’s most frequent mistake is hitting the shuttle out of bounds, which is consistent with his tactical style of Four-Corner, leading to shots landing near the boundary lines.

> **Summary**: From the rally length and energy expenditure chart, it is evident that Kento tends to win longer rallies. His playing style also puts considerable pressure on Axelsen's stamina, even in shorter rallies. By analyzing the tactical usage, we see that this is related to Momota's reliance on the Four-Corner, which focuses on front-court positioning and corner movement. In the 15th rally, for example, Momota’s net shots consistently increased his win rate. Furthermore, we found that his smash performance is significantly better on his forehand side. This trend is consistent when analyzing all of Momota’s past matches. As a result, opponents facing Momota in the future should ensure they have sufficient stamina and be prepared for his strategic maneuvering, especially for defending against forehand smashes.

> **Takeaway**: By importing match data and cross-referencing all three modules, we gain valuable insights into Momota's strengths. Applying these insights to training will likely increase the chances of success when facing Momota in future matches.

## Badminton Tactic Definition

### Definition
Based on the research "SWOT Analysis of Badminton Singles Tactics," we designed three main tactics for analysis: **Full-Court Pressure**, **Defensive Counterattack**, and **Four-Corner**.


![Screenshot](./Court%20Six%20Zones.png)

_Figure 1: Six zones defined from a subjective perspective_

| **Tactic**      | **Definition**                                         |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Full-Court Pressure | Player scores 1 point for a net shot or slice and 2 points for a smash; other shot types score 0 points. If the player accumulates 3 points in 2 shots, the interval is defined as Full-Court Pressure. If the player scores 4 points in 3 shots or 5 points in 4 shots, it also counts as Full-Court Pressure (excluding shots with 0 points at the beginning or end of the interval). |
| Defensive Counterattack    | After the opponent performs a smash, the player recieving enters a defensive state. If the player performs a smash, slice, or net shot from a defensive state, it is considered executing Defensive-Counterattack, starting from the beginning of the defensive state until the smash is completed.      |
| Four-Corner| The tactic is based on five subtactics: Forehand Lock, Backhand Lock, Front-Court Lock, Back-Court Lock, and Four-Corner Clear Drop. If one of the subtactics are executed, it's defined as Four-Corner.                                                                                              |


| **SubTactic in Four-Corner**        | **Definition**                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Four-Corner Clear Drop | Player’s shot placement differs from the previous one and lands in zones 1, 2, 5, or 6 (refer to Figure 1), scoring 1 point per occurrence. If the player accumulates 3 points in 4 shots and the last shot contributes a positive score, the interval is defined as Four-Corner Movement.                                                                                       |
| Forehand Lock        | Player’s shot lands in zones 3 or 5 in Figure 1, scoring 1 point per occurrence. If the player accumulates 2 points in 4 shots and both the first and last shots contribute positive scores, the interval is defined as Forehand Lock.                                                                                                                                       |
| Backhand Lock        | Player’s shot lands in zones 4 or 6 in Figure 1, scoring 1 point per occurrence. If the player accumulates 2 points in 4 shots and both the first and last shots contribute positive scores, the interval is defined as Backhand Lock.                                                                                                                                       |
| FrontCourt Lock     | Player’s shot lands in zones 1 or 2 in Figure 1, scoring 1 point per occurrence. If the player accumulates 2 points in 3 shots and both the first and last shots contribute positive scores, the interval is defined as FrontCourt Lock.                                                                                                                                      |
| BackCourt Lock      | Player’s shot lands in zones 5 or 6 in Figure 1 and is not a smash, scoring 1 point per occurrence. If the player accumulates 2 points in 4 shots and both the first and last shots contribute positive scores, the interval is defined as BackCourt Lock.                                                                                                                   |


### Tactic Classification Rules
We further refined the tactic classification into three levels, which are evaluated independently (except for level four):
- **Level 1**: Full-Court Pressure, Defensive Counterattack, Four-Corner
    - If none of the three tactics are used, it is classified as "No Tactic."
    - If the last two shots involve a Defensive Counterattack, it is classified as a Defensive Counterattack.
    - If only one of the three tactics are used, it is classified as that specific tactic. 
    - If neither defensive counterattack nor No Tactic is used, the tactic with the most recent execution between Four-Corner and Full-Court Pressure is chosen.
    - If Four-Corner and Full-Court Press both end at the same time, a smash in the final two shots classifies the interval as Full-Court Pressure; otherwise, it is classified as Four-Corner.
- **Level 2**: Forehand Lock, Backhand Lock
- **Level 3**: FrontCourt Lock, BackCourt Lock
    - At level 2 and 3, compare the total number of shots of each tactic, the tactic with higher number of shots is chosen for that level.
    - If the number of shots is equal, the tactics with most recent execution is chosen for that level.
- **Level 4**: Four-Corners Clear Drop
    - If the last three shots involve Four-Corners Clear Drop, and neither Level 2 nor Level 3 tactics were executed (or both were executed), it is classified as Four-Corners Clear Drop.


