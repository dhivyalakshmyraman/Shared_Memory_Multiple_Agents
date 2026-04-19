# Baseline Multi-Agent Test Report (No Memory)
==================================================

## Test Configuration
- Number of Agents: 6
- Total Tasks: 6
- Test Date: 2026-04-03 00:16:35

## Performance Metrics
- **Accuracy**: 33.33%
- **Average Execution Time**: 0.000s
- **Average Consensus Confidence**: 0.404
- **Agent Diversity**: 0.056

## Efficiency Metrics
- **Total Time**: 0.00s
- **Tasks per Second**: 1232.65
- **Average Agent Time**: 0.000s

## Task Breakdown
- Successful Tasks: 2/6
- Failed Tasks: 4/6

## Detailed Results
### Task 1: Janet has 3 apples and buys 5 more. She gives 2 away. How many apples does she have?
- **Ground Truth**: 6
- **Final Answer**: 6
- **Success**: True
- **Consensus**: 0.890
- **Time**: 0.000s

### Task 2: A store sells pencils for $2 each. If you buy 4 pencils, how much do you pay?
- **Ground Truth**: 8
- **Final Answer**: 8
- **Success**: True
- **Consensus**: 0.877
- **Time**: 0.000s

### Task 3: Sarah has 15 cookies. She eats 3 and gives 4 to her friend. How many cookies does she have left?
- **Ground Truth**: 8
- **Final Answer**: Unknown
- **Success**: False
- **Consensus**: 0.165
- **Time**: 0.000s

### Task 4: A train travels 60 miles per hour. How far will it travel in 3 hours?
- **Ground Truth**: 180
- **Final Answer**: Unknown
- **Success**: False
- **Consensus**: 0.165
- **Time**: 0.000s

### Task 5: The Eiffel Tower is located in which city and country?
- **Ground Truth**: Paris, France
- **Final Answer**: Unknown
- **Success**: False
- **Consensus**: 0.165
- **Time**: 0.000s

### Task 6: Which author wrote Hamlet and what nationality was he?
- **Ground Truth**: William Shakespeare, English
- **Final Answer**: Unknown
- **Success**: False
- **Consensus**: 0.165
- **Time**: 0.000s

## Agent Performance Summary
### conservative_agent_1
- Average Confidence: 0.375
- Success Rate: 33.33%

### creative_agent_1
- Average Confidence: 0.408
- Success Rate: 33.33%

### analytical_agent_1
- Average Confidence: 0.408
- Success Rate: 33.33%

### balanced_agent_1
- Average Confidence: 0.412
- Success Rate: 33.33%

### balanced_agent_2
- Average Confidence: 0.412
- Success Rate: 33.33%

### balanced_agent_3
- Average Confidence: 0.412
- Success Rate: 33.33%
