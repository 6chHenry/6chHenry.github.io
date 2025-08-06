# Python and Statistics for Financial Analysis

## Pandas Knowledge

1. What does shift() in `Pandas` mean?
`shift(-1)` means to move upwards by one row.

2. List Comprehension:
   `fb['Direction']=[1 if fb[ei,'PriceDiff'] > 0 else -1 for ei in fb.index]`

3. `fb['MovingAverage] = (fb + fb.shift(1) + fb.shift(2)) / 3` Move downwards to get the previous data.

## Financial Knowledge

PriceDiff = Close_Price_Of_Tomorrow - Close_Price_Of_Today

DailyReturn = PriceDiff / Close_Price_Of_Today

Direction :=

- PriceDiff $> 0  ====>$ Up 1
- PriceDiff $<= 0  ====>$ Down -1

MovingAverage = (Close_Price_Of_The_Day_Before_Yesterday + Close_Price_Of_Yesterday + Close_Price_Of_Today) / 3
