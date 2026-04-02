# Main Plan 


## Initial thoughts
1. Mothernet outputs mlp weights
2. Lets replace the MLP weights with tree weights.
3. Instead of an MLP, we output weights for an oblique tree (e.g. the weights for a SoftTree)

Given a dataset $\mathcal{D}_\text{train} = \{x_\text{train}, y_\text{train}\}$ and a depth $d$, we wanna predict $A, b$ which are the weights for a oblique tree. We use a depth of at most 8 and at most 100 features, so the output of this network will be $A \in \mathbb{R}^{255 \times 100}, b \in \mathbb{R}^{255}$. Depending on the depth $d$, we will max $A$ and $b$ appropriately. 

> Random Thought: Do we need to mask? We want to train for generalization so maybe its worth it to just let the network predict out the full tree, and if the weights shut down certain leaves thats fine. Oh wait, we want to mask/limit depth for interpretability. 

> Random Thought: Maybe we want multiple heads instead? Like instead of having 1 head and masking maybe we instead just output multiple trees? 
 
> Random Thought: Do we want the network to choose the tree? E.g. have D+1 heads, 1 for each depth and 1 for choosing final tree

Mothernet uses a low-rank matrix. Their MLP doesn't predict out $A$ it predicts $W_s \in \mathbb{R}^{r \times 100}$ and they have a learned matrix $W_c \in \mathbb{R}^{255\times r}$. $A$ is then equal to $W_c \times W_s$.  

> Implementation Detail: we need to confirm how Mothernet gets the bias vector from this. 

Also, since we are doing ICL and passing in the whole training set, we don't need to predict leaf labels, our inference pass can be output $A,b$, construct the tree, get soft leaf assignments on the training set, then use that to construct leaf labels. 

### Initial plan? 
* Maybe we can just fine-tune a mothernet for this first step. This might work IF the model size is the reason it took so long and needed so much RAM. 
* If the number of datasets is the issue, maybe we can instead using a smaller model with less priors  and see if the model even trains. 
* Any other ideas would be appreciated. 

### success criteria
* We do better than a soft tree trained specifically on the dataset. 
* We do better than RADDT trained specifically on the dataset. 
* We do better than tree alternating optimization trained specifically on the dataset. 
* We get very close to mother net. 
* We are limited to at most 256 classes e.g. the maximum number of leaves.
 
## Things we should improve on
* mother net attention is limite

## Novelty Criteria 
To make this a successful paper, we need to do more. Some ideas I have right now are: 
1. Special decoding strategy to allow for adaptive tree sizes so we are not limited to a specific tree size or number of classes. 
2. Interesting priors aka synthetic datasets that might help us better train trees. 
3. Eventually extending this to a *hard* somehow. Maybe not using RADDT but rather baking the hardness into the decoding. 
4. Sparsity *constraints:* If  we can figure out how to limit the number of features used in each node to at most $k$, that would be cool. 
   