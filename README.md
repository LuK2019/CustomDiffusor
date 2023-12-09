### Installation (on Mac)

Run the following commands in the repository. Skip the `brew` installation if python 3.8 is already on your systems.
```
brew install python@3.8
python3.8 --version
python3.8 -m venv customdiffusor
source customdiffusor/bin/activate
which python
python --version
pip3 install torch torchvision torchaudio
pip install 'diffusers[torch]' transformers
pip install matplotlib
```



ToDo's:
- Implement EMA for model weights during training
EMA Implementation: The use of EMA is a significant enhancement over a standard training loop. It often leads to better generalization in generative models.
Gradient Accumulation: This is particularly useful for training larger models or on hardware with limited memory.


References:

https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/reinforcement_learning_with_diffusers.ipynb#scrollTo=llzMmLk227jK

