### Coupon acceptance by customer

**Context**

This project is about Data analysis. A dataset related to the below use case has been taken to perform the analysis and in the process, all the concepts that have been learned till now have been put to use.

**Use Case**

Passengers are driven through a town and a coupon is delivered to their cell phone for a restaurant near where they are driving. Would the passenger accept that coupon and take a short detour to the restaurant? Would the passenger accept the coupon but use it on a subsequent trip? Would the passenger ignore the coupon entirely? What if the coupon was for a bar instead of a restaurant? What about a coffee house? Would the passenger accept a bar coupon with a minor passenger in the car? What about if it was just the passenger and his/her partner in the car? Would weather impact the rate of acceptance? What about the time of day?

**Analysis**

The data analysis was performed using a set of questions as mentioned below. The questions revolve around the possibility of accepting the coupon in the various scenarios. 

To explore the data, knowledge gained till now on plotting, statistical summaries, and visualization using Python was performed.

**Different scenarios**

1. & 2. Analyze the data for missing and problematic data.

> The different functions were used for this purpose after reading the file. 
> data.head() and data.info() to view the data columns and data types
> For loop was used to concisely format the list of the columns and the unique values used in each of the columns
> describe() was used to understad the stats on the numerical data
> isnull().sum() was used to see the blank values in each of the columns

3. Decide what to do about your missing data -- drop, replace, other...

> From the analysis it was clear that 'car' did not have any values for 12576 out of 12684 rows. This column can definitely be ignored for analysis
> Similarly, Bar, CoffeeHouse, CarryAway, Restaurant20To50 and RestaurantLessThan20 too did not have values for some of the rows. As these are not numerical, they can be ignored while performing the analysis on those rows. All the numerical data was filled

4. What proportion of the total observations chose to accept the coupon?

> A pie chart was plotted to view the % of coupons accepted vs not accepted and it was observed that 56.8% of the time the coupons were accepted

5. Use a bar plot to visualize the `coupon` column

> Analyzing the data for the 'coupon' column, and plotting a bar graph, it is observed that the most frequent coupon type is Coffee House, followed by Restaurant(<20).

6. Use a histogram to visualize the temperature column

> Since the temperature column contains only three unique values (30F, 55F, and 80F), the histogram acts as a frequency bar chart for these categories.  A histogram's purpose is to group continuous data. Since the data points are only at 30, 55, and 80, the histogram will show three tall bars with empty space (bins with zero count) in between.

***The next set of scenarios related to Bar coupons only***

7.1 & 7.2 Creating the dataset with only the Bar coupons and understand the proportion accepted

> A Pie chart was plotted to see that the accepted persentage was around 41%

7.3 Compare the acceptance rate between those who went to a bar 3 or fewer times a month to those who went more

> Low frequency bar visitors (<=3/month) and High frequency bar visitors (>3/month) was defined to calculate the acceptance % for each of the categories. 2 Pie charts were plotted for these 2 categories. It is observed that the acceptance rate was less, around 37.1%, for the Low frequency bar visitors. For the High frequency bar visitors, the acceptance was much higher at 73.2%, which is almost twice higher that the lower category

7.4 Compare the acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others.  Is there a difference?

> The comparison reveals a substantial difference in the 'Bar' coupon acceptance rate between the two groups. The Target Segment (drivers who go to a bar more than once a month AND are over the age of 25) shows an acceptance rate more than double that of the All Others segment. Yes, there is a significant difference. This specific segment of drivers is highly engaged and far more likely to accept a bar coupon than the general population of bar coupon recipients
>
> A bar grpah was drawn to depict the % acceptance for these 2 groups.

7.5 Use the same process to compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry

> The comparison shows a dramatic difference in the 'Bar' coupon acceptance rate between the two groups. The Target Segment (drivers who go to a bar more than once a month, had passengers that were not a kid, and had occupations other than farming, fishing, or forestry) is more than twice as likely to accept the coupon compared to the 'All Others' group. This segment represents a high-value customer group for bar promotions.
>
> The bar chart visually compares the acceptance rate (proportion of accepted coupons) for the two segments

7.6 Compare the acceptance rates between those drivers who:
- go to bars more than once a month, had passengers that were not a kid, and were not widowed *OR*
- go to bars more than once a month and are under the age of 30 *OR*
- go to cheap restaurants more than 4 times a month and income is less than 50K.

> The Combined Target Group has an acceptance rate of 59.14%, which is significantly higher (nearly double) the acceptance rate of 29.40% for the 'All Others' group. This combined targeting strategy effectively identifies drivers who are much more likely to redeem the bar coupon.
>
> The bar chart  visually represents the difference in the acceptance rate between the two groups.

7.7 Based on these observations, what do you hypothesize about drivers who accepted the bar coupons?

> Based on the previous detailed analyses, the core hypothesis about drivers who accepted the bar coupons is:
>
>***Hypothesis***
>
>Drivers who accepted the bar coupons are primarily frequent and established patrons (those who visit bars often and are older, more stable adults) who are likely visiting the bar in an adult social context (with friends or a partner, not with kids).
>
>This strategy maximizes coupon usage by targeting people who are already likely customers.
>
>***Supporting Graphs***
>
>The three bar charts support this hypothesis by visualizing the acceptance rate across the three key variables: Bar Frequency, Age, and Passenger Type.
>
>1. Acceptance Rate vs. Bar Frequency
>This graph provides the strongest evidence. There is a direct, strong positive correlation between how often a driver goes to a bar and their acceptance rate.
>    - Frequent Patrons (4~8, gt8): Acceptance rates are 73.47% to 78.00%.
>    - Non-Patrons (never): Acceptance rate drops to 18.80%.
>
>2. Acceptance Rate vs. Driver Age
>The hypothesis suggests older, established adults are more likely to accept. The data shows acceptance is highest for the youngest age groups, but the rates for those aged 30 and older are also significant:
>    - Ages 21 and 26: Highest acceptance rates (approx. 48% to 50%).
>    - Ages 30 and above: Acceptance rates stabilize between 29% and 44%, showing that drivers outside the 21-26 range still show substantial interest. The dip for ages 36 and 46 suggests complex life stage factors (e.g., family responsibilities) that the coupon marketers might be missing, but the overall rate for older drivers is still high enough to be valuable.
>
>3. Acceptance Rate vs. Passenger Type (Social Context)
>The hypothesis states that the bar coupons are more likely accepted in an adult social context.
>    - Adult Social Context (Friend(s)): Highest acceptance rate at 56.19%.
>    - Non-Social Context (Alone, Partner): Acceptance rates are lower (approx. 38% to 40%).
>    - Family Context (Kid(s)): Lowest acceptance rate at 20.69%, confirming that offers are less effective when children are present.

