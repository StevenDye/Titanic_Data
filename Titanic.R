# Load packages
library(ggplot2) # Visulaization
library(ggthemes) # Visualization
library(scales) # Visulaization
library(dplyr) # data manipulation
library(mice) # imputation
library(randomForest) # Classification Algorithm
library(stringr) # Character Manipulation

# Load raw data
train <- read.csv("input/train.csv", header = TRUE)
test <- read.csv("input/test.csv", header = TRUE)

full <- bind_rows(train, test) # bind training & test data

# Check data
str(full)


###########################
# Feature Engineering
###########################


# Titles and  Last Names
########################

# Split the name
name.splits <- str_split(full$Name, ",")
name.splits[1]

# Get Last names
last.names <- sapply(name.splits, "[",1)
last.names[1:10]

# Add last names to dataframe
full$Last.name <- last.names

# Now make a list of titles
name.splits <- str_split(sapply(name.splits, "[", 2), " ")
titles <- sapply(name.splits, "[", 2)
unique(titles)

# To prevent over fitting, re-map titles to be more exact
titles[titles %in% c("Dona.", "Mme.", "the", "Lady.")] <- "Mrs."
titles[titles %in% c("Ms.", "Mlle.")] <- "Miss."
titles[titles %in% c("Jonkheer.", "Don.", "Col.", "Capt.", "Major.", "Dr.", "Rev.", "Sir.")] <- "Mr."
table(titles)

# Make title a factor
full$Title <- as.factor(titles)

# One of the Dr.'s is female. Let's fix their title.
indexes <- which(full$Title == "Mr." & full$Sex == "female")
full$Title[indexes] <- "Mrs."

# Any other gender slip-ups?
length(which(full$Sex == "female" &
               (full$Title == "Master." |
                  full$Title == 'Mr.')))

# Show title counts by sex
table(full$Sex, full$Title)


# Family Size
##############

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Create a family variable 
full$Family <- paste(full$Last.name, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Discretize family size
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# Show family size by survival using a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)


# Cabin Size
##############
# This variable appears to have a lot of missing values
full$Cabin[1:28]

# The first character is the deck. For example:
strsplit(full$Cabin[2], NULL)[[1]]

# Create a Deck variable. Get passenger deck A - F:
full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x,NULL)[[1]][1]))


##################
# Missing Data
##################


# Sensible value imputation
###########################


# Passengers 62 and 830 are missing Embarkment
full[c(62, 830), 'Embarked']

# We will infer their values for embarkment based on the present data that may be relevant:
# passenger class and fare.

# We see that the paid
full[c(62), 'Fare']
full[c(830), 'Fare']

# And were in class
full[c(62), 'Pclass']
full[c(830), 'Pclass']

# So where did they emark?

# Get rid of missing passenger IDs
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

# Use gplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             color='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# Since they both paid $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] <- 'C'

# Any missing information in Fare?
full[1044,]
# This is a third class passenger who departed from Southampton ('S'). Let's visualize Fares among
# all others sharing their class and embarkment (n = 494).

ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

# Replace missing fare value with median fare for class/embarkment
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

# Find Avg. Fare by diving Fare by the amount of people who share the same ticket number
########################################################################################

# Calculate Average Fare Feature
ticket.party.size <- rep(0, nrow(full))
avg.fare <- rep(0.0, nrow(full))
tickets <- unique(full$Ticket)

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(full$Ticket == current.ticket)
  current.avg.fare <- full[party.indexes[1],"Fare"] / length(party.indexes)
  
  for (k in 1:length(party.indexes)) {
    ticket.party.size[party.indexes[k]] <- length(party.indexes)
    avg.fare[party.indexes[k]] <- current.avg.fare
  }
}

full$Ticket.Party.Size <- ticket.party.size
full$Avg.Fare <- avg.fare


# Predictive Imputation
########################


# Show number of missing Age values
sum(is.na(full$Age))

# Make variable factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked', 'Avg.Fare',
                 'Title','Surname','Family','FsizeD', 'Ticket.Party.Size')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family',
                                            'Surname','Survived')], method='rf') 

# Save the complete output 
mice_output <- complete(mice_mod)

# Compare the results with the original distribution of passenger ages to ensure nothing
# has gone completely awry.

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# This looks good, so let's replace the age vector in the original data with the 
# output from the mice model.

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

# We'll look at the relationship between age & survival
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss.'] <- 'Mother'

# Show counts
table(full$Mother, full$Survived)

# Finish by factorizing our two new factor variables
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

md.pattern(full)


####################
# Prediction
####################


# Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]


# Build the model
###################


# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
#rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + SibSp + Parch + 
#                           Fare + Embarked + Title + Age +  
#                           FsizeD + Child + Mother,
#                         data = train)


# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Avg.Fare + Ticket.Party.Size + Title +
                           Child + Mother,
                         data = train)


# FsizeD and Emabarked are bad

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Variable Importance
#####################


# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


##########################
# Prediction
##########################


# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'output/R_RF_Titanic_Submission.csv', row.names = FALSE)

