# 🧠 TRANSFORMERS Y ATTENTION MECHANISMS

## Guía de Estudio para Large Language Models

<img width="824" height="690" alt="image" src="https://github.com/user-attachments/assets/95b96ee6-c55e-4df8-b6da-d5b867060c75" />

---

---

## 🎯 **CONCEPTOS CLAVE**

### 🔥 **¿Qué es un Transformer?**

> Arquitectura fundamental de los LLMs modernos (GPT, Claude, BERT)
> 

**Componentes principales:**

- 🧩 **Tokenizer** - Convierte texto a números
- 🎯 **Embeddings** - Representación vectorial inteligente
- 📍 **Positional Encoding** - Información del orden
- 🧠 **Stack de Transformer Blocks** - Procesamiento principal
- 🎪 **LM Head** - Predicción del siguiente token

---

## **ARQUITECTURA COMPLETA DEL TRANSFORMER LLM**

```
📝 Input Text
    ↓
🔢 TOKENIZER
    ↓
🧩 EMBEDDINGS + POSITIONAL ENCODING
    ↓
🔄 STACK OF TRANSFORMER BLOCKS
   │
   ├── 🎯 Self-Attention (relaciones entre palabras)
   └── 🧠 Feed Forward NN (procesamiento complejo)
    ↓
🎪 LM HEAD (Language Model Head)
    ↓
🎲 DECODING STRATEGY
    ↓
📤 Next Output Token
```

---

## 🔢 Proceso de funcionamiento

## **Paso 1: Tokenización**

```
Texto: "Diego eats pizza"
↓
Tokens: ["Diego", "eats", "pizza"]

```

## **Paso 2: Word Embeddings**

- **Qué es:** Vector numérico que representa cada token
- **Propósito:** Capturar características semánticas
- **Ejemplo:**
    
    ```
    "pizza" → [0.2, -0.1, 0.8, 0.5, ...]          
                ↑     ↑     ↑    ↑        
               comida dulce calor Italia
    ```
    

❌ **Problema con números fijos:**

`"love" = 5
"hate" = 6
→ El modelo pensaría que "hate" es similar a "love" (números cercanos)`

## **Paso 3: Positional Encoding**

⚠️ **¿Por qué es crucial?**

| Oración A | Oración B |
| --- | --- |
| "Diego eats pizza" → **Yum** | "Pizza eats Diego" → **Run** |
- **Función:** Distinguir el orden de las palabras
- **Resultado:** Embeddings + información posicional

## **Paso 4: Attention**

🎯 **Determina qué palabras se relacionan entre sí**

**Ejemplo práctico:**

> "La pizza acaba de salir del horno, estaba rica"
> 

**Attention identifica:** "estaba" → se refiere a "pizza" (no a "horno")

---

## 📏 **CONTEXT LENGTH (Longitud del Contexto)**

### **Definición Simple:**

> Cuántas palabras puede "recordar" el modelo en total
> 

### **Ejemplo Fácil:**

`Pregunta: "I love pizza"    →  3 palabras
Respuesta: "yummy"          →  1 palabra

Context Length = 4 palabras en total`

### **¿Por qué importa?**

- 🧠 **Memoria limitada:** El modelo no puede recordar textos infinitos
- 📝 **Conversaciones largas:** Se "olvida" de lo que dijiste al principio
- 📊 **Límites actuales:**
    - ChatGPT: ~4,000 palabras
    - Claude: ~100,000 palabras
    - Algunos modelos nuevos: +1,000,000 palabras

---

## 🔄 **PROCESAMIENTO AUTOREGRESIVO**

### **¿Qué es Autoregresivo?**

> Cada token generado se agrega al input para generar el siguiente
> 

### **Ejemplo - Traducción Inglés → Holandés:**

`Step 1: "I love llamas" → "ik"
Step 2: "I love llamas ik" → "hou" 
Step 3: "I love llamas ik hou" → "van"
Step 4: "I love llamas ik hou van" → "lama's"

Resultado Final: "ik hou van lama's"`

### **🔑 Características Clave:**

- 🎯 **Secuencial:** Una palabra a la vez
- 🔄 **Iterativo:** El output se convierte en input
- 🧠 **Contextual:** Cada nueva palabra considera las anteriores

---

## 🎪 **LM HEAD (Language Model Head)**

### **Función Principal:**

> Se encarga de **generar el siguiente token** asignando puntajes a los tokens más probables del vocabulario.
> 

Proceso:
`Transformer Output → LM Head → Probabilidades de Vocabulario`

### **Ejemplo de Output:**

```
TokenId      Token       Probs
0            !           0.01%
1            "           0.03%
.
.
.
1002         Dear        40%
5000         Zyzzyua     1%
```

---

## 🎲 **ESTRATEGIAS DE DECODIFICACIÓN**

### 1️⃣ **Greedy Decoding**

`🎯 Siempre elige el token con mayor probabilidad`

- **Configuración:** `temperature = 0`
- **Ventaja:** Determinístico y consistente
- **Desventaja:** Puede ser repetitivo

### 2️⃣ **Sampling con Temperatura**

`🎲 Agrega aleatoriedad controlada`

- **Configuración:** `temperature > 0`
- **Proceso:** Selecciona entre los top_p tokens más probables
- **Ventaja:** Más creativo y variado
- **Control:** `top_p` determina cuántos tokens considerar

---

## 🧮 **TIPOS DE ATTENTION**

### 1️⃣ **Self-Attention**

```
🔄 BIDIRECCIONAL (ve pasado y futuro)

```

**Características:**

- ✅ Analiza **todas** las palabras de la secuencia
- ✅ Incluye la palabra que se está evaluando
- 🎯 **Uso:** Encoders (BERT)

**Fórmula:**

```
Attention(Q,K,V) = SoftMax(QK^T/√d_k) × V

```

### 2️⃣ **Masked Self-Attention**

```
⬅️ UNIDIRECCIONAL (solo ve el pasado)

```

**Características:**

- 🚫 **NO** ve palabras futuras
- ✅ Solo considera palabra actual y anteriores
- 🎯 **Uso:** Decoders (GPT, Claude, Gemini)

**Fórmula:**

```
Attention(Q,K,V,M) = SoftMax((QK^T/√d_k) + M) × V

```

---

## 🔑 **COMPONENTES DE LA FÓRMULA**

| Símbolo | Significado | Descripción |
| --- | --- | --- |
| **Q** | Query | 🔍 "¿Qué estoy buscando?" |
| **K** | Key | 🗝️ Clave/identificador de cada palabra |
| **V** | Value | 💎 Valor asociado con la palabra |
| **M** | Mask | 🎭 Bloquea información futura |
| **d_k** | Dimensión | 📏 Tamaño del embedding |

### 🧮 **Cálculo de Q, K, V:**

```
Q = Embedding × W_Q  (pesos de Query)
K = Embedding × W_K  (pesos de Key)
V = Embedding × W_V  (pesos de Value)
M = Embedding × W_M  (pesos de Mask)

```

⚡ **Los pesos W se aprenden durante el entrenamiento y son fijos en inferencia**

---

## 🏗️ **ARQUITECTURAS PRINCIPALES**

### 🔍 **ENCODERS (BERT)**

```
📖 Comprensión bidireccional

```

- **Attention:** Self-Attention
- **Propósito:** Entender contexto completo
- **Aplicaciones:** Clasificación, Q&A, análisis

### 🗣️ **DECODERS (GPT, Claude, Gemini)**

```
✍️ Generación secuencial

```

- **Attention:** Masked Self-Attention
- **Propósito:** Generar texto palabra por palabra
- **Aplicaciones:** Chat, escritura, código

---

## 🚀 **MULTI-HEAD ATTENTION**

### 💡 **¿Por qué múltiples cabezas?**

- 📝 **Oraciones largas y complejas** necesitan múltiples perspectivas
- 🎯 Cada "cabeza" se especializa en diferentes tipos de relaciones

### 🔄 **Proceso:**

1. **Múltiples capas** de attention en paralelo
2. **Concatenación** de resultados
3. **Proyección lineal** para ajustar dimensiones

```
[Head1, Head2, Head3, ...] → Concat → Linear → Output

```

⚠️ **Importante:** Si el output > embedding size → se agrega capa adicional para reducir el size del output

---

## 🧠 **FEED FORWARD NEURAL NETWORK**

> Red neuronal compleja que procesa la información de attention
> 

### **Función:**

- 🎯 **Transforma** las representaciones de attention
- 🧮 **Aplica transformaciones no-lineales**
- 🎪 **Prepara información** para el LM Head
- 📊 **Múltiples capas** con activaciones (ReLU, GELU)

---

## 🎯 **TIPOS DE EMBEDDINGS GENERADOS**

### 1️⃣ **Word Embeddings**

- 🎯 **Clusters de palabras similares**
- Ejemplo: ["gato", "perro", "mascota"] se agrupan

### 2️⃣ **Context-Aware Embeddings**

- 🎯 **Clusters de oraciones/documentos**
- Ejemplo: Diferentes significados de "banco" según contexto
