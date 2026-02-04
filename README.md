# Brain Tumor Classification using Vision Transformer

A deep learning project that classifies brain tumor types using a Vision Transformer (ViT) model. The system can identify four different tumor types from MRI images.

## 🧠 Tumor Classes

- **Glioma Tumor**
- **Meningioma Tumor** 
- **No Tumor**
- **Pituitary Tumor**

## 📁 Project Structure

```
Transformer/
├── dataset/
│   ├── Training/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── no_tumor/
│       └── pituitary_tumor/
├── brain_tumor_classification.py    # Training script
├── best_model.pth                   # Trained model weights
├── confusion_matrix.png             # Evaluation visualization
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pillow tqdm
```

### 2. Prepare Dataset

Organize your MRI images in the following structure:

```
dataset/
├── Training/
│   ├── glioma_tumor/     # Training images for glioma
│   ├── meningioma_tumor/ # Training images for meningioma
│   ├── no_tumor/         # Training images for no tumor
│   └── pituitary_tumor/  # Training images for pituitary
└── Testing/
    ├── glioma_tumor/     # Testing images for glioma
    ├── meningioma_tumor/ # Testing images for meningioma
    ├── no_tumor/         # Testing images for no tumor
    └── pituitary_tumor/  # Testing images for pituitary
```

### 3. Train the Model

```bash
python brain_tumor_classification.py
```

The training will:
- Load and preprocess the dataset
- Train a Vision Transformer model
- Save the best model as `best_model.pth`
- Generate a confusion matrix

## 🛠️ Model Architecture

### Vision Transformer (ViT) Configuration
- **Base Model**: `google/vit-base-patch16-224`
- **Input Size**: 224x224 pixels
- **Patch Size**: 16x16 pixels
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12

### Custom Classification Head
- **Linear Layer**: Maps ViT output to 4 tumor classes
- **Output**: 4-dimensional logits for tumor classification

## 📊 Training Configuration

```python
# Hyperparameters
batch_size = 16
num_epochs = 20
learning_rate = 3e-5
weight_decay = 0.01
img_size = 224
```

### Data Augmentation
- Random horizontal flip
- Random rotation (±15 degrees)
- ImageNet normalization

## 📈 Performance Metrics

The model evaluation includes:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1 score
- **Confusion Matrix**: Visual representation of predictions

## 🔧 Technical Details

### Image Preprocessing
1. **Resize**: 224x224 pixels
2. **Normalization**: ImageNet standards
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### Device Support
- **CUDA**: Automatically uses GPU if available
- **CPU**: Fallback to CPU processing

### Model Saving
- Best model saved based on test accuracy
- PyTorch `.pth` format
- Contains model state dict only

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'torch'**
   ```bash
   pip install torch torchvision
   ```

2. **CUDA out of memory**
   - Reduce batch size in Config class
   - Use CPU instead: `device = torch.device("cpu")`

3. **Corrupted model file**
   - Delete `best_model.pth` and retrain
   - Check if training completed successfully

4. **Dataset not found**
   - Verify dataset directory structure
   - Check image file formats (.jpg, .png, .bmp)

### Python Version Compatibility
- **Recommended**: Python 3.8+
- **Tested**: Python 3.12
- **Not compatible**: Python 3.13 (use 3.12 instead)

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=8.0.0
tqdm>=4.60.0
numpy>=1.21.0
```

## 🎯 Model Performance

Expected performance metrics (based on training):
- **Test Accuracy**: ~90-95%
- **Precision**: ~0.90-0.95
- **Recall**: ~0.90-0.95
- **F1-Score**: ~0.90-0.95

*Note: Actual performance may vary based on dataset quality and size*

## 🔬 Medical Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📝 License

This project is provided for educational purposes. Please ensure compliance with relevant medical data regulations and ethical guidelines when using patient data.

## 🤝 Contributing

Feel free to contribute improvements:
- Add new tumor types
- Implement different architectures
- Improve data augmentation
- Add evaluation metrics

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify dataset structure
3. Ensure all dependencies are installed
4. Check Python version compatibility
=======
<div align="center">
  <img src="https://media.giphy.com/media/dWesBcTLavkZuG35MI/giphy.gif" width="75%"/>
</div>

<h1 align="center">
  👋 Hi! I’m Paarth<img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="28">
</h1>
<h2 align="center">A passionate developer, tech explorer, and lifelong learner. I'm driven by curiosity and inspired by solving real-world problems through code.</h2>

## 🚀 About Me
- 🧠 Computer Science student focused on **algorithms**, **AI**, and **software engineering**
- 💻 Languages: **Python**, **C++**, **JavaScript**
- 🌐 Tech Stack: **Pygame**, **Tkinter**, **React.js**, **Node.js**, **MongoDB**, **Firebase**
- 🎯 Interests: **Game Development**, **AI/ML**, **Blockchain in Education**, **Open Source**

  
<p>
  <a href="https://vaunt.dev">
    <img src="https://api.vaunt.dev/v1/github/entities/Paarth01/contributions?format=svg" width="350" title="Includes public contributions"/>
  </a>
</p>
<h2>Leetcode Info<h2>
<p>
  <img  align=top flex-grow=1 src="https://leetcard.jacoblin.cool/Paarth_agarwal?theme=dark&font=Nunito&ext=heatmap" />  
</p>
<p align="left"> <img src="https://komarev.com/ghpvc/?username=paarth01&label=Profile%20views&color=0e75b6&style=flat" alt="paarth01" /> </p>

## 🏆 GitHub Trophies
<p align="left"> <a href="https://github.com/ryo-ma/github-profile-trophy"><img src="https://github-profile-trophy.vercel.app/?username=paarth01" alt="paarth01" /></a> </p>

<p align="left"> <a href="https://twitter.com/" target="blank"><img src="https://img.shields.io/twitter/follow/?logo=twitter&style=for-the-badge" alt="" /></a> </p>

<img align="right" alt="Coding" width="350" src="https://i.pinimg.com/originals/e8/d0/f1/e8d0f1794e2520ac2367c1d21c0966e9.gif">

### 🔭 I’m currently working on  
- **Pac-Man Game** using advanced pathfinding algorithms (Dijkstra and A*) to enhance ghost AI and gameplay mechanics.  
- **Virtual Mouse** controlled by AI hand gestures for intuitive and touchless computer interaction.
  
### 🌱 I’m currently learning  
Modern **Web Development** techniques and deepening my mastery of **Data Structures & Algorithms** to build efficient, scalable applications.
  
### 👯 I’m looking to collaborate on  
Innovative projects involving game development, AI/ML, and intuitive user interfaces—always open to new challenges and ideas.

### 🤝 I’m looking for help with  
Feedback on AI integration, user experience design, and project scalability to improve my current projects.

### 💬 Ask me about  
Game development, pathfinding algorithms, AI-powered gesture controls, frontend development, C++, Python, and data structures.

### 📫 How to reach me  
Email: [agl.paarth2006@gmail.com](mailto:agl.paarth2006@gmail.com)

### ⚡ Fun fact  
Honey never spoils — archaeologists have found edible honey over 3,000 years old!

## 🛠️ Current Projects

- 🔍 **Shortest Path Finder**  
  Interactive tool using A* and Dijkstra’s algorithm with a Tkinter GUI and Matplotlib-based graph visualizations.

- 👾 **Pac-Man Reimagined**  
  A 2D Pac-Man clone in Pygame enhanced with custom power-ups, ghost AI, and pathfinding mechanics.

- 🖱️ **Gesture-Controlled Mouse**  
  AI-driven system that maps hand gestures to mouse actions (clicks, scrolls) using computer vision.

- 🔗 **SkillChain** *(In Progress)*  
  A blockchain-powered educational platform rewarding learners via smart contracts and verifiable credentials.

<div align="center">
  <img src="https://media.giphy.com/media/LOEgEYZHyGbXY8KgA3/giphy.gif" width = "1rem"  height = "1rem" />
</div>

## 🌐 Socials :
[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://instagram.com/paarth_0101)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)]()
[![LinkedIn](https://img.shields.io/badge/Linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/paarth-agarwal-1b595a288)
[![LeetCode](https://img.shields.io/badge/LeetCode-000000.svg?style=for-the-badge&logo=LeetCode&logoColor=#d16c06)](https://leetcode.com/u/Paarth_agarwal/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836.svg?style=for-the-badge&logo=gmail&logoColor=white)](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox)

## 💻Tech Stack <img src = "https://media2.giphy.com/media/QssGEmpkyEOhBCb7e1/giphy.gif?cid=ecf05e47a0n3gi1bfqntqmob8g9aid1oyj2wr3ds3mg700bl&rid=giphy.gif" width = 32px>
![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white) 
![CPP](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white) 
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)  
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![Express](https://img.shields.io/badge/Express%20js-000000?style=for-the-badge&logo=express&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB) 
![Node](https://img.shields.io/badge/Node%20js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%238511FA.svg?style=for-the-badge&logo=bootstrap&logoColor=white)
![Styled Components](https://img.shields.io/badge/styled--components-DB7093?style=for-the-badge&logo=styled-components&logoColor=white) ![Adobe](https://img.shields.io/badge/adobe-%23FF0000.svg?style=for-the-badge&logo=adobe&logoColor=white) 
![Adobe After Effects](https://img.shields.io/badge/Adobe%20After%20Effects-9999FF.svg?style=for-the-badge&logo=Adobe%20After%20Effects&logoColor=white) 
![Adobe Illustrator](https://img.shields.io/badge/adobe%20illustrator-%23FF9A00.svg?style=for-the-badge&logo=adobe%20illustrator&logoColor=white) ![Adobe Lightroom](https://img.shields.io/badge/Adobe%20Lightroom-31A8FF.svg?style=for-the-badge&logo=Adobe%20Lightroom&logoColor=white) 
![Adobe Photoshop](https://img.shields.io/badge/adobe%20photoshop-%2331A8FF.svg?style=for-the-badge&logo=adobe%20photoshop&logoColor=white) 
![Adobe Premiere Pro](https://img.shields.io/badge/Adobe%20Premiere%20Pro-9999FF.svg?style=for-the-badge&logo=Adobe%20Premiere%20Pro&logoColor=white)
![Canva](https://img.shields.io/badge/Canva-%2300C4CC.svg?style=for-the-badge&logo=Canva&logoColor=white)

# 📊 GitHub Stats:
<br>
<div align=center>
  <img width=390 src="https://streak-stats.demolab.com/?user=Paarth01&count_private=true&theme=react&border_radius=10" alt="streak stats"/>
  <img width=390 src="https://github-readme-stats.vercel.app/api?username=Paarth01&show_icons=true&theme=react&rank_icon=github&border_radius=10" alt="readme stats" />
  <img width=325 align="center" src="https://github-readme-stats.vercel.app/api/top-langs/?username=Paarth01&hide=HTML&langs_count=8&layout=compact&theme=react&border_radius=10&size_weight=0.5&count_weight=0.5&exclude_repo=github-readme-stats" alt="top langs" />
</div>

  <br/>
