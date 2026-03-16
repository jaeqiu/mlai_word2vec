## Project

By learning per paragraph-batch, we are applying the learning rate step agnostically of the amount of words in the paragraph.
 "I am wing" will result in 4 instances I -> am, am -> I, am -> wing, wing -> am
 "I am wing it is nice to meet you" will result in more instances
 By applying the same total learning step for both, we are learning "I -> am" more heavily in the short sentence than the long one.
 This is undesirable, since we want to learn context-words based on center-words, agnostically of the words outside of the context.
