# What is SIGMOS?

SIGMOS is a Deep Learning Model that estimates the quality of speech signals. It is an 
objective metric that approximates subjective human ratings. It is based on ITU-T P.804 
standard for audio quality evaluation. The model is trained on over 20,000 speech files and 
predicts the Signal and Overall speech quality as well as five speech quality dimensions: 
Noisiness, Coloration, Discontinuity, Reverb and Loudness.

# What are SIGMOS’s intended use(s)?

Researchers in the field of speech quality and communication networks can use SIGMOS to 
conduct studies and experiments related to speech quality assessment. It provides a 
standardized and automated way to measure and analyze speech quality. SIGMOS can also 
assist in identifying the causes of speech quality degradation, such as noise, distortion, or 
poor network conditions. 

# What are the limitations of SIGMOS?

While SIGMOS is a valuable tool for speech quality assessment, it does have some 
limitations. Here are some of the key limitations:
- SIGMOS is designed to assess the quality of speech in various scenarios, but it may 
not be suitable for assessing the quality of other audio types, such as music or non-speech sounds.
- While SIGMOS predicts various speech quality dimensions (e.g., Noisiness, 
Coloration, Discontinuity, Loudness, Reverb), there may be other quality dimensions 
or factors that are not captured by the model.
- The training data for SIGMOS is limited to the audio clips and quality ratings used in 
its training dataset. It may not account for the full diversity of audio conditions and 
real-world scenarios, potentially resulting in suboptimal performance for audio 
outside the training data's scope.
- While for SIGMOS training we used at least 4 ratings per clip, the quality ratings 
assigned by humans in the training data may be influenced by individual biases, 
expectations, or preferences. Biased ratings can affect the model's ability to provide 
objective quality assessments.