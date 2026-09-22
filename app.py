import io
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from fpdf import FPDF


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dense Embedding-Based Semantic Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_FILE = Path("data/documents.csv")
MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
THEORY_FIGURE_DIR = BASE_DIR / "assets" / "theory"


# ============================================================
# HTML RENDERING HELPER
# ============================================================
#
# st.markdown(..., unsafe_allow_html=True) runs the string through a
# Markdown parser first. Markdown ends an HTML block at a blank line,
# and treats any line indented 4+ spaces as a literal code block -- which
# is what previously leaked raw <div> text into the top-left of the page.
#
# render_html() flattens the fragment to a single unindented line before
# it reaches the parser, so it is emitted as real HTML every time.

def render_html(markup):
    cleaned = " ".join(
        line.strip()
        for line in markup.strip().splitlines()
        if line.strip()
    )

    st.markdown(cleaned, unsafe_allow_html=True)


def to_html_paragraphs(text):
    blocks = [
        block.strip().replace("\n", " ")
        for block in text.strip().split("\n\n")
        if block.strip()
    ]

    return "".join(f"<p>{block}</p>" for block in blocks)


def escape_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* ---------------- GENERAL PAGE ---------------- */

    .stApp { background: white; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .block-container {
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }

    /* ---------------- TOP HEADER ---------------- */

    .top-header {
        height: 105px;
        display: flex;
        align-items: center;
        background: white;
        border-bottom: 1px solid #dddddd;
        padding: 10px 25px;
    }

    .menu-symbol {
        font-size: 34px;
        color: #777777;
        margin-right: 25px;
        margin-top: 2px;
    }

    .logo-main {
        font-size: 30px;
        font-weight: 500;
        color: #248bc4;
        line-height: 30px;
    }

    .logo-sub {
        font-size: 18px;
        font-weight: 400;
        color: #8bc53f;
        margin-left: 5px;
    }

    .logo-caption {
        font-size: 10px;
        color: #888888;
        margin-top: 5px;
    }

    .header-spacer { flex: 1; }

    .rating {
        color: #f4bd00;
        font-size: 25px;
        letter-spacing: 2px;
        margin-right: 25px;
    }

    .header-button {
        background: #2696d2;
        color: white;
        border-radius: 17px;
        padding: 13px 24px;
        margin-left: 10px;
        font-size: 15px;
        display: inline-block;
        white-space: nowrap;
    }

    .orange-line {
        height: 8px;
        background: #f47721;
        width: 100%;
    }

    /* ---------------- BREADCRUMB ---------------- */

    .breadcrumb {
        color: #2d86c5;
        font-size: 24px;
        padding: 26px 12px 20px 12px;
        font-weight: 400;
    }

    /* ---------------- LEFT NAVIGATION ---------------- */

    .st-key-left_nav_panel {
        border-right: 1px solid #e5e5e5;
        padding: 5px 25px 25px 32px;
        min-height: 650px;
    }

    .left-navigation-title {
        color: #2d86c5;
        font-size: 21px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .st-key-left_nav_panel [class*="st-key-nav_"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #315a83 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 10px 0px !important;
        margin: 0px !important;
    }

    .st-key-left_nav_panel [class*="st-key-nav_"] button:hover {
        color: #f47721 !important;
        background: transparent !important;
        text-decoration: underline;
    }

    .st-key-left_nav_panel [class*="st-key-nav_"] button:focus {
        box-shadow: none !important;
    }

    /* ---------------- MAIN CONTENT ---------------- */

    .st-key-main_content_panel { padding: 5px 48px 35px 48px; }

    .experiment-heading {
        color: #2d9bd3;
        font-size: 31px;
        font-weight: 400;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 18px;
        margin-bottom: 30px;
    }

    .content-heading {
        color: #2d9bd3;
        font-size: 28px;
        font-weight: 400;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 12px;
        margin-top: 24px;
        margin-bottom: 18px;
    }

    .content-subheading {
        color: #2d9bd3;
        font-size: 23px;
        font-weight: 400;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .content-text {
        color: #222222;
        font-size: 17px;
        line-height: 1.7;
    }

    .aim-list {
        color: #222222;
        font-size: 17px;
        line-height: 1.8;
    }

    .info-box {
        border-left: 4px solid #2d9bd3;
        background: #f5fbff;
        padding: 15px 20px;
        margin: 15px 0px;
        font-size: 16px;
        color: #333333;
    }

    .result-box {
        border-left: 4px solid #f47721;
        background: #fff8f1;
        padding: 15px 20px;
        margin: 15px 0px;
        font-size: 16px;
        color: #333333;
    }

    /* ---------------- PIPELINE DIAGRAM ---------------- */

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin: 12px 0px 18px 0px;
    }

    .pipeline-stage {
        background: #f5fbff;
        border: 1px solid #cfe6f5;
        color: #23648c;
        border-radius: 4px;
        padding: 8px 14px;
        font-size: 14px;
        font-weight: 600;
    }

    .pipeline-arrow {
        color: #f47721;
        font-size: 18px;
        font-weight: 700;
    }

    .stage-label {
        display: inline-block;
        background: #f47721;
        color: white;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 3px;
        margin-bottom: 6px;
    }

    /* ---------------- RESULT CARDS ---------------- */

    .result-card {
        border: 1px solid #d8d8d8;
        border-left: 5px solid #2696d2;
        border-radius: 5px;
        padding: 18px;
        margin: 15px 0px;
        background: #ffffff;
    }

    .result-rank {
        color: #f58220;
        font-size: 15px;
        font-weight: bold;
    }

    .result-title {
        color: #2696d2;
        font-size: 21px;
        margin-top: 5px;
    }

    .result-category {
        color: #777777;
        font-size: 14px;
        margin: 5px 0px 12px 0px;
    }

    .result-content {
        color: #333333;
        font-size: 15px;
        line-height: 1.6;
    }

    .result-score {
        color: #198754;
        font-weight: bold;
        margin-top: 12px;
    }

    .result-meta {
        color: #888888;
        font-size: 13px;
        margin-top: 6px;
    }

    /* ---------------- STREAMLIT CONTROLS ---------------- */

    .stButton > button {
        background: #2696d2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #197db6;
        color: white;
    }

    button[data-baseweb="tab"] {
        color: #2d86c5;
        font-size: 16px;
    }

    /* ---------------- FOOTER ---------------- */

    .vlab-footer {
        background: #111111;
        color: white;
        display: flex;
        justify-content: space-around;
        padding: 13px;
        font-size: 14px;
        margin-top: 25px;
    }

    </style>
    """
)


EXPERIMENT_TITLE = "Dense Embedding-Based Semantic Search"

# ============================================================
# DOCUMENT CORPUS
# ============================================================
#
# The corpus lives here as a compact {category: [(title, content), ...]}
# mapping and is expanded into a full DataFrame by build_default_documents().
# The same data is exported to data/documents.csv, which the app prefers
# when present -- so the collection can be edited without touching code.

CORPUS = {
    "Artificial Intelligence": [
        ("Introduction to Artificial Intelligence", "Artificial intelligence studies how machines can perform tasks that normally require human intelligence, including reasoning, perception, planning and decision making under uncertainty."),
        ("Search Algorithms in AI", "Uninformed and informed search strategies such as breadth-first search, depth-first search and A star explore a state space to find a path from an initial state to a goal state."),
        ("Knowledge Representation", "Knowledge representation encodes facts about the world in a form a machine can reason over, using logic, semantic networks, frames and ontologies."),
        ("Expert Systems", "An expert system captures human expertise as a rule base and uses an inference engine to draw conclusions, offering explanations for the advice it produces."),
        ("Propositional and Predicate Logic", "Logical inference lets an agent derive new statements from known ones using rules such as modus ponens, resolution and unification over quantified expressions."),
        ("Intelligent Agents", "An intelligent agent perceives its environment through sensors and acts through actuators, selecting actions that maximise a performance measure over time."),
        ("Planning and Scheduling", "Automated planning constructs a sequence of actions that transforms an initial world state into a goal state while respecting preconditions and resource constraints."),
        ("Constraint Satisfaction Problems", "Constraint satisfaction formulates a problem as variables, domains and constraints, solved by backtracking search combined with constraint propagation and heuristics."),
        ("Game Playing and Adversarial Search", "Minimax search with alpha-beta pruning evaluates game trees for two player games, trading search depth against the accuracy of the evaluation function."),
        ("Ethics and Explainability in AI", "Responsible artificial intelligence addresses fairness, accountability, transparency and the ability to explain why an automated system produced a particular decision."),
    ],
    "Machine Learning": [
        ("Introduction to Machine Learning", "Machine learning is a branch of artificial intelligence that allows computers to learn patterns from data and make predictions without being explicitly programmed for each task."),
        ("Supervised Learning Algorithms", "Supervised learning uses labelled training data to learn a mapping from inputs to outputs, covering algorithms such as regression, decision trees and support vector machines."),
        ("Unsupervised Learning and Clustering", "Unsupervised learning discovers structure in unlabelled data, grouping similar observations using techniques such as k-means, hierarchical clustering and density based methods."),
        ("Linear and Logistic Regression", "Linear regression fits a continuous target as a weighted sum of features, while logistic regression models class probability using the sigmoid function and cross entropy loss."),
        ("Decision Trees and Random Forests", "Decision trees split data on feature thresholds to form interpretable rules, and random forests average many randomised trees to reduce variance and improve generalisation."),
        ("Support Vector Machines", "Support vector machines find the hyperplane with the maximum margin between classes and use kernel functions to separate data that is not linearly separable."),
        ("Overfitting and Regularization", "A model that memorises training noise generalises poorly, so regularisation, early stopping and cross validation are used to control model complexity."),
        ("Feature Engineering and Selection", "Feature engineering transforms raw attributes into informative inputs through scaling, encoding and aggregation, while selection removes redundant or irrelevant variables."),
        ("Model Evaluation Metrics", "Accuracy, precision, recall, F1 score and the area under the ROC curve measure different aspects of classifier quality, especially on imbalanced datasets."),
        ("Ensemble Learning Methods", "Ensemble methods such as bagging, boosting and stacking combine several weak learners into a stronger predictor with lower bias or lower variance."),
    ],
    "Deep Learning": [
        ("Introduction to Deep Learning", "Deep learning uses artificial neural networks with multiple layers to learn hierarchical representations and complex patterns directly from large datasets."),
        ("Artificial Neural Networks", "A feedforward neural network passes inputs through weighted connections and nonlinear activation functions to compute an output used for classification or regression."),
        ("Backpropagation and Gradient Descent", "Backpropagation computes the gradient of the loss with respect to every weight using the chain rule, and gradient descent updates the weights to reduce the loss."),
        ("Convolutional Neural Networks", "Convolutional networks apply learned filters across an image to detect local patterns such as edges and textures, using pooling layers to build translation tolerant features."),
        ("Recurrent Neural Networks and LSTM", "Recurrent networks maintain a hidden state across time steps for sequential data, while LSTM and GRU cells use gates to preserve information over long sequences."),
        ("Transformer Architecture", "The transformer replaces recurrence with self attention, letting every token attend to every other token in parallel and scaling efficiently to very large corpora."),
        ("Attention Mechanisms", "Attention computes a weighted combination of value vectors using the similarity between queries and keys, allowing a model to focus on the most relevant parts of its input."),
        ("Transfer Learning and Fine Tuning", "Transfer learning reuses a model pretrained on a large corpus and adapts it to a smaller downstream task by fine tuning some or all of its parameters."),
        ("Regularization in Neural Networks", "Dropout, batch normalisation, weight decay and data augmentation reduce overfitting in deep networks and stabilise the optimisation process."),
        ("Generative Models and Autoencoders", "Autoencoders compress input into a latent code and reconstruct it, while generative adversarial networks and diffusion models learn to synthesise realistic new samples."),
    ],
    "Natural Language Processing": [
        ("Introduction to Natural Language Processing", "Natural language processing enables computers to understand, process and generate human language using computational linguistics and machine learning methods."),
        ("Tokenization and Text Preprocessing", "Preprocessing converts raw text into tokens through sentence splitting, lowercasing, stopword removal, stemming and lemmatisation before any model is applied."),
        ("Part of Speech Tagging", "Part of speech tagging assigns a grammatical category such as noun, verb or adjective to each token, usually with sequence models or neural taggers."),
        ("Named Entity Recognition", "Named entity recognition locates and classifies spans of text that refer to people, organisations, locations, dates and other predefined entity types."),
        ("Word Embeddings", "Word embeddings such as Word2Vec, GloVe and FastText map words to dense vectors so that words used in similar contexts lie close together in the vector space."),
        ("Sentence Embeddings and Sentence-BERT", "Sentence-BERT fine tunes a transformer with a siamese structure so that whole sentences map to vectors whose cosine similarity reflects semantic similarity."),
        ("Language Models", "A language model assigns probabilities to sequences of tokens, evolving from n-gram counts to neural and transformer based models trained on very large corpora."),
        ("Machine Translation", "Neural machine translation encodes a source sentence into a continuous representation and decodes it into a target language, largely replacing phrase based statistical systems."),
        ("Text Classification and Sentiment Analysis", "Text classification assigns documents to categories, and sentiment analysis determines whether the opinion expressed in a text is positive, negative or neutral."),
        ("Question Answering Systems", "Question answering systems retrieve or generate an answer to a natural language question, either by extracting a span from a passage or by generating free text."),
    ],
    "Information Retrieval": [
        ("Introduction to Information Retrieval", "Information retrieval is the process of finding relevant information from a large collection of documents in response to a user's stated information need."),
        ("Inverted Index Construction", "An inverted index maps every term to a posting list of documents that contain it, enabling fast lookup and intersection during query processing."),
        ("Boolean Retrieval Model", "The boolean model treats a query as a logical expression over terms and returns the set of documents that exactly satisfy the expression, without ranking."),
        ("Vector Space Model and TF-IDF", "The vector space model represents documents as weighted term vectors, where TF-IDF raises the weight of terms that are frequent locally but rare across the collection."),
        ("BM25 Ranking Function", "BM25 is a probabilistic ranking function that scores documents using term frequency saturation and document length normalisation, and remains a strong lexical baseline."),
        ("Dense Retrieval and Semantic Search", "Dense retrieval encodes queries and documents as embeddings and ranks by vector similarity, retrieving relevant results even when no query term appears in the document."),
        ("Cosine Similarity in Retrieval", "Cosine similarity measures the angle between two vectors, giving a score near one for texts with similar meaning and near zero for unrelated texts."),
        ("Evaluation of Retrieval Systems", "Retrieval quality is measured with precision, recall, mean average precision and normalised discounted cumulative gain computed against human relevance judgements."),
        ("Query Expansion and Relevance Feedback", "Query expansion adds related terms from a thesaurus, an embedding space or the top ranked results in order to reduce vocabulary mismatch."),
        ("Hybrid and Re-ranking Pipelines", "Hybrid search merges lexical and dense candidate lists, then applies a cross encoder re-ranker to the shortlist to improve the quality of the final ordering."),
    ],
    "Computer Networks": [
        ("Introduction to Computer Networks", "Computer networks allow multiple devices to communicate and share resources using agreed communication protocols such as the TCP/IP protocol suite."),
        ("OSI Reference Model", "The OSI model organises communication into seven layers, from physical transmission up to the application layer, each offering services to the layer above."),
        ("TCP and UDP Transport Protocols", "TCP provides reliable, ordered, connection oriented delivery with flow and congestion control, while UDP offers lightweight connectionless datagram service."),
        ("IP Addressing and Subnetting", "IP addressing identifies hosts on a network, and subnetting divides an address block into smaller networks using a mask to separate network and host portions."),
        ("Routing Algorithms", "Routing protocols use distance vector or link state algorithms to compute forwarding paths, with OSPF operating inside an autonomous system and BGP between them."),
        ("Network Security Protocols", "TLS, IPSec and SSH protect data in transit using encryption, message authentication codes and certificate based authentication of the communicating endpoints."),
        ("Wireless and Mobile Networks", "Wireless networks transmit over a shared radio medium and must handle interference, attenuation, mobility and handover between access points or base stations."),
        ("Domain Name System", "The domain name system resolves human readable names into IP addresses through a hierarchy of authoritative and caching resolvers distributed worldwide."),
        ("Congestion Control and Quality of Service", "Congestion control adjusts sending rate in response to loss or delay signals, while quality of service mechanisms prioritise traffic that is sensitive to latency."),
        ("Software Defined Networking", "Software defined networking separates the control plane from the data plane, allowing a central controller to program forwarding behaviour through open interfaces."),
    ],
    "Operating Systems": [
        ("Introduction to Operating Systems", "An operating system manages computer hardware and software resources and provides common services and abstractions to application programs."),
        ("Process Management and Scheduling", "The scheduler decides which ready process runs next using policies such as round robin, shortest job first and multilevel feedback queues."),
        ("Threads and Concurrency", "Threads share an address space within a process, enabling parallelism but requiring synchronisation to avoid race conditions on shared data."),
        ("Deadlock Detection and Prevention", "Deadlock occurs when processes hold resources and wait circularly, and is handled by prevention, avoidance with the banker's algorithm, or detection and recovery."),
        ("Memory Management and Paging", "Paging divides memory into fixed size frames and maps virtual pages onto them, using a page table and translation lookaside buffer for fast address translation."),
        ("Virtual Memory and Page Replacement", "Virtual memory lets a process use more address space than physical memory, with replacement policies such as LRU and clock deciding which page to evict on a fault."),
        ("File Systems and Directory Structures", "A file system organises data into files and directories and tracks allocation with inodes, extents or allocation tables while enforcing access permissions."),
        ("Input Output and Device Management", "The operating system mediates access to devices through drivers, interrupts and buffering, hiding hardware differences behind a uniform interface."),
        ("Synchronization Primitives", "Semaphores, mutexes, condition variables and monitors coordinate concurrent threads and protect critical sections from simultaneous access."),
        ("Virtualization and Containers", "Hypervisors run multiple virtual machines on one host, while containers share the host kernel and isolate processes using namespaces and control groups."),
    ],
    "Database Management": [
        ("Introduction to Database Management Systems", "A database management system is software used to store, organise, retrieve and manage structured data efficiently while enforcing integrity and access control."),
        ("Relational Data Model", "The relational model represents data as tables of tuples, with keys expressing identity and relationships and relational algebra defining the query operations."),
        ("Structured Query Language", "SQL is a declarative language for defining, querying and modifying relational data, combining selection, projection, joins, grouping and aggregation."),
        ("Normalization and Database Design", "Normalisation decomposes relations to remove redundancy and update anomalies, progressing through first, second, third and Boyce-Codd normal forms."),
        ("Indexing and Query Optimization", "Indexes such as B-trees and hash indexes speed up lookups, and the query optimiser chooses an execution plan using cost estimates and table statistics."),
        ("Transactions and ACID Properties", "A transaction is an atomic unit of work that preserves consistency, runs in isolation from concurrent transactions and durably records its effects."),
        ("Concurrency Control", "Locking protocols, timestamp ordering and multiversion concurrency control allow transactions to run in parallel while preserving serialisable behaviour."),
        ("NoSQL Databases", "NoSQL systems store key-value pairs, documents, wide columns or graphs, trading strict schemas and joins for horizontal scalability and flexible data models."),
        ("Vector Databases", "Vector databases store numerical embeddings and support approximate nearest neighbour search for recommendation systems, semantic search and retrieval augmented generation."),
        ("Data Warehousing and OLAP", "A data warehouse integrates historical data into star or snowflake schemas so that analytical queries can aggregate across many dimensions efficiently."),
    ],
    "Cloud Computing": [
        ("Introduction to Cloud Computing", "Cloud computing provides on-demand access to computing resources such as servers, storage, databases and networking over the internet with pay per use billing."),
        ("Service Models IaaS PaaS SaaS", "Infrastructure, platform and software as a service differ in how much of the stack the provider manages and how much control remains with the customer."),
        ("Deployment Models", "Public, private, hybrid and community clouds differ in ownership and tenancy, balancing cost and elasticity against control and regulatory requirements."),
        ("Virtual Machines and Hypervisors", "Cloud providers allocate virtual machines through hypervisors that partition physical servers and isolate tenants from one another."),
        ("Containers and Orchestration", "Container orchestration platforms such as Kubernetes schedule containers across a cluster and handle scaling, service discovery and rolling updates."),
        ("Serverless Computing", "Serverless platforms run functions in response to events and bill only for execution time, removing the need to provision or manage servers explicitly."),
        ("Cloud Storage Systems", "Object, block and file storage services offer durable, replicated storage with different consistency guarantees and access patterns."),
        ("Load Balancing and Auto Scaling", "Load balancers distribute requests across healthy instances, while auto scaling adds or removes capacity according to demand or scheduled policies."),
        ("Cloud Security and Identity Management", "Cloud security relies on the shared responsibility model, identity and access management policies, encryption at rest and in transit, and continuous auditing."),
        ("Edge and Fog Computing", "Edge computing moves processing close to where data is produced to reduce latency and bandwidth use, complementing centralised cloud data centres."),
    ],
    "Cybersecurity": [
        ("Introduction to Cyber Security", "Cyber security protects computer systems, networks and data from unauthorised access, attacks and threats to confidentiality, integrity and availability."),
        ("Symmetric and Asymmetric Cryptography", "Symmetric ciphers use one shared key for encryption and decryption, while public key cryptography uses a key pair to enable secure exchange and digital signatures."),
        ("Hash Functions and Digital Signatures", "Cryptographic hash functions produce a fixed length digest that is hard to invert, and digital signatures bind a message to the identity of its signer."),
        ("Authentication and Access Control", "Authentication verifies identity through passwords, tokens or biometrics, while access control models decide which subjects may act on which objects."),
        ("Network Attacks and Defences", "Firewalls, intrusion detection systems and segmentation defend against scanning, spoofing, denial of service and lateral movement inside a network."),
        ("Malware Analysis", "Malware analysis studies viruses, worms, trojans and ransomware through static inspection and dynamic execution in a sandbox to derive detection signatures."),
        ("Web Application Security", "Injection, cross site scripting, broken authentication and insecure deserialisation are common web vulnerabilities mitigated by validation and secure defaults."),
        ("Public Key Infrastructure", "A public key infrastructure issues, distributes and revokes certificates through certificate authorities so that parties can trust one another's public keys."),
        ("Security Auditing and Penetration Testing", "Penetration testing simulates an attacker to find exploitable weaknesses, while auditing verifies that controls match policy and regulatory requirements."),
        ("Privacy and Data Protection", "Data protection combines anonymisation, minimisation, encryption and consent management to limit exposure of personal information."),
    ],
    "Data Science": [
        ("Introduction to Data Science", "Data science combines statistics, programming and machine learning to extract useful insights from structured and unstructured data."),
        ("Exploratory Data Analysis", "Exploratory analysis summarises distributions, correlations and outliers using descriptive statistics and visualisation before any model is fitted."),
        ("Data Cleaning and Preprocessing", "Cleaning handles missing values, duplicates, inconsistent units and encoding errors so that downstream analysis is not distorted by data quality problems."),
        ("Statistical Inference and Hypothesis Testing", "Hypothesis testing quantifies whether an observed effect is likely to arise by chance, reporting p-values, confidence intervals and effect sizes."),
        ("Data Visualization", "Effective visualisation chooses encodings that match the data type and the question, revealing trends, comparisons and distributions without distorting scale."),
        ("Dimensionality Reduction", "Principal component analysis, t-SNE and UMAP project high dimensional data into fewer dimensions for visualisation, denoising or faster computation."),
        ("Time Series Analysis", "Time series methods model trend, seasonality and autocorrelation with techniques such as ARIMA, exponential smoothing and sequence neural networks."),
        ("Big Data Processing Frameworks", "Distributed frameworks such as Hadoop and Spark partition data across a cluster and execute parallel transformations over the partitions."),
        ("Recommendation Systems", "Recommenders combine collaborative filtering, content based similarity and embeddings to suggest items a user is likely to find relevant."),
        ("Experimentation and A/B Testing", "A/B testing randomly assigns users to variants and compares a target metric, requiring sufficient sample size and careful handling of multiple comparisons."),
    ],
    "Blockchain": [
        ("Introduction to Blockchain Technology", "A blockchain is an append only ledger replicated across many nodes, where blocks of transactions are linked by cryptographic hashes to make tampering detectable."),
        ("Distributed Ledger and Consensus", "Consensus protocols such as proof of work, proof of stake and practical byzantine fault tolerance let mutually distrusting nodes agree on a single ledger state."),
        ("Smart Contracts", "Smart contracts are programs stored on a blockchain that execute automatically when conditions are met, with their results recorded on the shared ledger."),
        ("Cryptographic Hashing in Blockchain", "Each block stores the hash of its predecessor and a Merkle root of its transactions, so any modification invalidates every subsequent block."),
        ("Public and Private Blockchains", "Public chains allow anyone to join and validate, while permissioned chains restrict participation to known members and favour throughput over open access."),
        ("Cryptocurrency and Digital Wallets", "A wallet stores private keys that authorise transfers, and the network verifies signatures before including a transaction in a block."),
        ("Blockchain Scalability Solutions", "Sharding, payment channels and rollups increase throughput by moving work off the main chain while still anchoring final state to it."),
        ("Blockchain in Supply Chain", "Recording custody events on a shared ledger gives supply chain participants a tamper evident audit trail of provenance and handling."),
        ("Security Issues in Blockchain", "Majority attacks, smart contract bugs, key theft and front running are the main practical risks in blockchain deployments."),
        ("Decentralized Applications", "Decentralised applications combine smart contracts on chain with off chain interfaces and storage to deliver services without a central operator."),
    ],
    "Distributed Systems": [
        ("Introduction to Distributed Systems", "A distributed system is a collection of independent computers that appears to its users as a single coherent system despite partial failures and network delays."),
        ("Distributed Consensus Algorithms", "Paxos and Raft allow a replicated set of servers to agree on a log of operations even when some members crash or messages are delayed."),
        ("CAP Theorem and Consistency Models", "The CAP theorem states that a partitioned system must choose between consistency and availability, motivating eventual and causal consistency models."),
        ("Replication and Fault Tolerance", "Replication stores copies of data on several nodes so that service continues after failures, at the cost of keeping replicas synchronised."),
        ("Remote Procedure Calls and Messaging", "Remote procedure calls make a network request look like a local call, while message queues decouple producers from consumers and absorb load spikes."),
        ("Distributed File Systems", "Distributed file systems split files into chunks replicated across servers, with a metadata service tracking placement and handling recovery."),
        ("Clock Synchronization and Logical Time", "Physical clocks drift, so distributed systems use logical clocks and vector timestamps to order events consistently across nodes."),
        ("MapReduce Programming Model", "MapReduce expresses a computation as map and reduce functions that the framework runs in parallel over partitioned data with automatic fault recovery."),
        ("Microservices Architecture", "Microservices split an application into independently deployable services that communicate over the network, improving autonomy but adding operational complexity."),
        ("Load Balancing in Distributed Systems", "Load balancers spread requests across replicas using round robin, least connections or consistent hashing to avoid hotspots and keep latency stable."),
    ],
    "Computer Architecture": [
        ("Introduction to Computer Architecture", "Computer architecture describes the organisation of processors, memory and interconnects, and the instruction set interface exposed to software."),
        ("Instruction Set Architecture", "An instruction set defines the operations, registers and addressing modes a processor supports, with RISC and CISC representing different design philosophies."),
        ("Pipelining and Hazards", "Pipelining overlaps instruction stages to raise throughput, while data, control and structural hazards are resolved with forwarding, stalls and branch prediction."),
        ("Cache Memory and Hierarchy", "Caches exploit temporal and spatial locality, and the memory hierarchy trades capacity against latency from registers through caches to main memory and storage."),
        ("Memory Organization", "Main memory is organised into banks and channels, and interleaving together with burst transfers improves the effective bandwidth seen by the processor."),
        ("Parallel Processing and Multicore", "Multicore processors run several threads simultaneously, requiring cache coherence protocols and careful synchronisation to scale performance."),
        ("GPU Architecture", "Graphics processors devote most of their area to many simple cores executing the same instruction over different data, which suits dense linear algebra workloads."),
        ("Instruction Level Parallelism", "Superscalar issue, out of order execution and speculation extract parallelism from a single instruction stream without changing the programming model."),
        ("Input Output Organization", "Processors communicate with peripherals through memory mapped registers, interrupts and direct memory access to avoid busy waiting."),
        ("Performance Measurement and Benchmarking", "Performance is evaluated with clock rate, cycles per instruction and benchmark suites, while Amdahl's law bounds the speedup achievable from partial optimisation."),
    ],
    "Software Engineering": [
        ("Introduction to Software Engineering", "Software engineering applies systematic and measurable approaches to the development, operation and maintenance of software systems."),
        ("Software Development Life Cycle", "The development life cycle moves through requirements, design, implementation, testing, deployment and maintenance, in a waterfall or iterative arrangement."),
        ("Agile Methodologies", "Agile methods such as Scrum and Kanban deliver working software in short iterations, adapting scope through continuous feedback from stakeholders."),
        ("Requirements Engineering", "Requirements engineering elicits, analyses, specifies and validates what a system must do, distinguishing functional from quality requirements."),
        ("Software Design Patterns", "Design patterns describe reusable solutions to recurring design problems, including creational, structural and behavioural families such as factory, adapter and observer."),
        ("Software Testing Techniques", "Unit, integration, system and acceptance testing use black box and white box techniques to expose defects before software reaches production."),
        ("Version Control and Collaboration", "Distributed version control tracks changes as commits on branches, letting teams work in parallel and merge their work with a reviewable history."),
        ("Continuous Integration and Deployment", "Continuous integration builds and tests every change automatically, and continuous deployment pipelines release validated builds with minimal manual effort."),
        ("Software Quality Assurance", "Quality assurance combines reviews, static analysis, coverage measurement and defect tracking to keep quality visible throughout development."),
        ("Software Maintenance and Refactoring", "Maintenance covers corrective, adaptive and perfective changes, while refactoring restructures code without altering behaviour to reduce technical debt."),
    ],
}

SOURCE_TYPES = [
    "Lecture Notes",
    "Reference Textbook",
    "Survey Paper",
    "Technical Manual",
    "Course Handout",
]

# ============================================================
# QUESTION BANKS
# ============================================================
#
# q(question, correct_option, *three_distractors)
# The correct option is written first for readability; render_quiz()
# rotates the option order per question so the answer is not always
# in the same position.

def q(question, correct, *distractors):
    return {
        "question": question,
        "options": [correct, *distractors],
        "answer": correct,
    }


PRETEST_QUESTIONS = [
    q("What is the main purpose of a sentence embedding?",
      "To convert text into a numerical vector",
      "To delete all words from a sentence",
      "To convert text into an image",
      "To sort documents alphabetically"),
    q("Which similarity measure is used in this experiment?",
      "Cosine similarity", "Euclidean distance only", "Manhattan distance only", "Jaccard index only"),
    q("What does Top-K represent?",
      "The number of documents returned", "The number of input characters",
      "The number of model layers", "The embedding dimension"),
    q("Which model family is used to generate the embeddings here?",
      "SentenceTransformer", "Linear Regression", "Decision Tree", "K-Means clustering"),
    q("A cosine similarity score close to 1 generally indicates:",
      "High semantic similarity", "No relationship", "An empty document", "A failed search"),
    q("Semantic search retrieves documents based on:",
      "The meaning of the text", "Only exact keyword matches",
      "Alphabetical order of titles", "The file size of the document"),
    q("The main limitation of keyword search is that it:",
      "Fails when different words express the same meaning",
      "Cannot read text files", "Requires a GPU", "Works only on images"),
    q("What is the embedding dimension of the all-MiniLM-L6-v2 model?",
      "384", "12", "1024", "50000"),
    q("A dense vector is best described as:",
      "A fixed-length list of numbers where most values are non-zero",
      "A list containing mostly zeros", "A plain text string", "A table of keywords"),
    q("Why is an embedding index built before searching?",
      "So document embeddings are computed once and reused for every query",
      "To delete duplicate documents", "To translate the documents", "To compress the images"),
    q("For L2-normalised vectors, the cosine similarity is equal to:",
      "Their dot product", "Their sum", "Their difference", "Their product of lengths"),
    q("NLP stands for:",
      "Natural Language Processing", "Network Layer Protocol",
      "Numerical Linear Programming", "Neural Logic Processing"),
    q("Tokenization is the process of:",
      "Splitting text into smaller units such as words or subwords",
      "Encrypting the text", "Deleting punctuation only", "Sorting words by length"),
    q("Stopwords are:",
      "Very common words that often carry little meaning",
      "Words that stop a program", "Misspelled words", "Words unique to one document"),
    q("In information retrieval, a corpus means:",
      "The collection of documents being searched",
      "A single query", "The ranking function", "The similarity threshold"),
    q("The query embedding must be generated using:",
      "The same model used for the documents", "Any random model",
      "A keyword counter", "A decision tree"),
    q("Ranking in a search system means:",
      "Ordering documents by their relevance score",
      "Deleting irrelevant documents", "Renaming the documents", "Splitting long documents"),
    q("A similarity threshold is used to:",
      "Reject documents whose score is too low",
      "Increase the embedding size", "Change the model weights", "Speed up file uploads"),
    q("Which Python library provides the SentenceTransformer class?",
      "sentence-transformers", "matplotlib", "openpyxl", "requests"),
    q("The cosine similarity formula divides the dot product by:",
      "The product of the vector magnitudes", "The sum of the vectors",
      "The number of documents", "The embedding dimension"),
    q("A transformer is:",
      "A neural network architecture based on attention",
      "A type of database index", "A file compression format", "A clustering algorithm"),
    q("The advantage of using a pretrained model is that:",
      "It already captures language patterns learned from large corpora",
      "It never needs any input", "It removes the need for a computer", "It stores the documents"),
    q("Normalising embeddings to unit length mainly helps to:",
      "Make cosine similarity depend only on direction",
      "Reduce the number of documents", "Increase the vocabulary", "Remove stopwords"),
    q("The dot product of two identical unit vectors equals:",
      "1", "0", "-1", "The vector dimension"),
    q("Which of these is a metadata field in our document collection?",
      "Category", "Learning rate", "Batch normalisation", "Dropout"),
    q("TF-IDF is a technique used in:",
      "Lexical keyword weighting", "Image compression",
      "Memory paging", "Process scheduling"),
    q("Compared with a sparse TF-IDF vector, a dense embedding is:",
      "Lower dimensional with mostly non-zero values",
      "Always larger in size", "Made only of zeros", "Always binary"),
    q("Embeddings can capture meaning because similar texts:",
      "Are placed close together in the vector space",
      "Contain the same file extension", "Have equal word counts", "Are alphabetically adjacent"),
    q("An inverted index maps:",
      "Terms to the documents that contain them",
      "Documents to their file size", "Queries to users", "Vectors to images"),
    q("Precision measures:",
      "The fraction of retrieved documents that are relevant",
      "The fraction of relevant documents retrieved", "The time taken per query", "The vector length"),
    q("Recall measures:",
      "The fraction of relevant documents that were retrieved",
      "The fraction of retrieved documents that are relevant", "The index build time", "The batch size"),
    q("A cosine similarity near 0 between a query and a document means:",
      "They are semantically unrelated", "They are identical",
      "They are opposites in meaning", "The model has crashed"),
    q("Cosine similarity values lie in the range:",
      "-1 to 1", "0 to 100", "1 to 384", "-384 to 384"),
    q("Which of the following is NOT a stage of the search pipeline?",
      "Disk defragmentation", "Query embedding", "Cosine similarity", "Ranking"),
    q("If the similarity threshold is set very high, the system will:",
      "Return very few or no documents", "Return every document",
      "Crash immediately", "Increase the embedding dimension"),
    q("Increasing Top-K generally:",
      "Returns more documents, including less relevant ones",
      "Improves the model accuracy", "Reduces the embedding dimension", "Deletes old trials"),
    q("Document embeddings are stored in the index so that:",
      "They are not recomputed for every new query",
      "They can be printed on paper", "The model can be deleted", "The query becomes shorter"),
    q("Batch size during encoding controls:",
      "How many documents are encoded together in one pass",
      "The number of categories", "The similarity threshold", "The number of search results"),
    q("Which column is essential in an uploaded CSV document dataset?",
      "content", "colour", "password", "resolution"),
    q("Which file formats can be uploaded in this simulation?",
      "CSV, JSON and TXT", "MP3 and WAV", "EXE and DLL", "PNG and JPG"),
    q("Encoding documents on a GPU instead of a CPU usually:",
      "Reduces the index build time", "Increases the embedding dimension",
      "Changes the similarity formula", "Deletes the index"),
    q("Query latency in this experiment is reported in:",
      "Milliseconds", "Kilobytes", "Degrees", "Pixels"),
    q("Which chart type is used to compare similarity scores of results?",
      "Horizontal bar chart", "Pie chart", "Candlestick chart", "Gantt chart"),
    q("Compared with boolean search, semantic search:",
      "Ranks results by degree of relevance",
      "Returns only exact matches", "Cannot rank documents", "Ignores the query completely"),
    q("A vector database is designed to:",
      "Store embeddings and perform similarity search",
      "Store only images", "Replace the operating system", "Compile source code"),
    q("Using a different model for the query and for the documents would:",
      "Place them in incompatible vector spaces",
      "Improve the accuracy", "Halve the search time", "Have no effect at all"),
    q("L2 normalisation changes a vector so that:",
      "Its length becomes 1", "All its values become 0",
      "Its dimension doubles", "It becomes a string"),
    q("Cosine distance is usually defined as:",
      "1 minus the cosine similarity", "The sum of the two vectors",
      "The square of the similarity", "The vector dimension"),
    q("Does semantic search require the query words to appear in the document?",
      "No, related meaning is enough", "Yes, always",
      "Only for CSV documents", "Only when Top-K is 1"),
    q("The final output of this simulation is:",
      "A ranked list of documents with similarity scores",
      "A trained neural network", "A compressed archive", "An encrypted database"),
]


POSTTEST_QUESTIONS = [
    q("Which stage converts the document collection into the embedding index?",
      "Stage A, document indexing", "Stage B, query search",
      "The feedback form", "The PDF report generator"),
    q("Why is indexing separated from searching?",
      "Indexing is expensive and its result can be reused by every query",
      "Because queries cannot be embedded", "To reduce the number of documents",
      "Because the model changes for each query"),
    q("If 150 documents are indexed with a 384-dimensional model, the index matrix shape is:",
      "150 x 384", "384 x 384", "150 x 150", "1 x 384"),
    q("Adding new documents to the collection requires you to:",
      "Rebuild the embedding index", "Restart the operating system",
      "Change the similarity metric", "Retrain the transformer"),
    q("Self-attention in a transformer allows each token to:",
      "Attend to every other token in the sequence",
      "Ignore all other tokens", "Be processed strictly in order",
      "Be replaced by a stopword"),
    q("Sentence-BERT improves on plain BERT for similarity because it:",
      "Produces sentence vectors that can be compared directly with cosine similarity",
      "Uses no neural network", "Removes attention entirely", "Only works on single words"),
    q("Mean pooling over token embeddings is used to:",
      "Produce one fixed-length vector for a whole sentence",
      "Increase the sequence length", "Remove the attention weights", "Tokenize the input"),
    q("A siamese network architecture trains two encoders that:",
      "Share the same weights", "Use different vocabularies",
      "Never see the same data", "Produce vectors of different sizes"),
    q("Comparing a query against N documents by brute force costs:",
      "O(N) similarity computations", "O(1) computations",
      "O(N log N) disk writes", "O(N squared) computations"),
    q("Approximate nearest neighbour search is used when:",
      "The collection is too large for exact brute-force scanning",
      "The query is empty", "The embedding dimension is 1", "Only one document exists"),
    q("HNSW and IVF are examples of:",
      "Approximate nearest neighbour index structures",
      "Loss functions", "Tokenizers", "Activation functions"),
    q("Hybrid search combines:",
      "Lexical scores such as BM25 with dense embedding scores",
      "Two identical dense models", "Images and audio", "Two different databases only"),
    q("A cross-encoder re-ranker is typically applied:",
      "To a small shortlist returned by the first-stage retriever",
      "To the entire collection for every query", "Before tokenization", "Only during indexing"),
    q("Retrieval augmented generation uses semantic search to:",
      "Supply relevant context to a language model before it answers",
      "Train the language model from scratch", "Compress the vector index",
      "Replace the tokenizer"),
    q("Mean Reciprocal Rank rewards a system for:",
      "Placing the first relevant document as high as possible",
      "Returning as many documents as possible", "Having a low embedding dimension",
      "Using the least memory"),
    q("nDCG differs from precision because it:",
      "Accounts for the position and graded relevance of results",
      "Ignores relevance entirely", "Measures only index build time",
      "Counts the number of categories"),
    q("Precision@5 of 0.6 means that out of the top five results:",
      "Three are relevant", "Six are relevant", "All are relevant", "None are relevant"),
    q("Raising the similarity threshold typically:",
      "Increases precision and decreases recall",
      "Increases both precision and recall", "Decreases precision and increases recall",
      "Has no effect on either"),
    q("If a relevant document scores just below the threshold, it will be:",
      "Excluded from the results", "Promoted to rank one",
      "Re-embedded automatically", "Added to the index twice"),
    q("A query that returns many documents with nearly identical scores suggests:",
      "The query is generic and weakly discriminative",
      "The index is corrupted", "The model has 0 dimensions", "Top-K was set to 1"),
    q("Vocabulary mismatch is the problem where:",
      "The query and the relevant document use different words for the same idea",
      "The document has too many words", "Two documents have the same title",
      "The model file is missing"),
    q("Dense retrieval addresses vocabulary mismatch because it compares:",
      "Meaning in a shared vector space rather than surface terms",
      "Character counts", "File timestamps", "Document identifiers"),
    q("A weakness of dense retrieval compared with BM25 is:",
      "It can miss exact rare terms such as product codes",
      "It cannot rank documents", "It needs no model", "It always runs slower than disk search"),
    q("Out-of-domain queries perform worse in dense retrieval because:",
      "The encoder was trained on different kinds of text",
      "Cosine similarity stops working", "The threshold becomes negative",
      "The index deletes itself"),
    q("Chunking long documents before embedding is done because:",
      "A single vector cannot represent a very long text well",
      "Models cannot read text files", "Cosine similarity needs equal lengths",
      "Categories must be unique"),
    q("An overlapping window when chunking helps to:",
      "Avoid splitting a relevant passage across chunk boundaries",
      "Reduce the embedding dimension", "Increase the batch size",
      "Remove duplicate documents"),
    q("Embedding 150 short documents at batch size 32 requires approximately:",
      "5 batches", "150 batches", "1 batch", "32 batches"),
    q("The cosine similarity of a vector with itself is:",
      "1", "0", "-1", "Its dimension"),
    q("Two vectors pointing in exactly opposite directions have cosine similarity:",
      "-1", "0", "1", "384"),
    q("Cosine similarity ignores vector magnitude, which means:",
      "A long and a short text about the same topic can still score highly",
      "Long texts always win", "Short texts always win", "Magnitude must be zero"),
    q("Euclidean distance on L2-normalised vectors is:",
      "Monotonically related to cosine similarity",
      "Completely unrelated to cosine similarity", "Always zero", "Always negative"),
    q("The curse of dimensionality refers to:",
      "Distances becoming less discriminative as dimensions grow",
      "Models having too few parameters", "Files being too small",
      "Queries being too short"),
    q("Dimensionality reduction such as PCA is used in this simulation to:",
      "Visualise the embedding space in two dimensions",
      "Improve retrieval accuracy", "Compress the document text",
      "Encrypt the embeddings"),
    q("In the embedding scatter plot, documents of the same category cluster because:",
      "Their contents are semantically similar",
      "They were uploaded together", "They share a document id prefix",
      "They have equal word counts"),
    q("Quantising embeddings from float32 to int8 mainly:",
      "Reduces memory usage with a small accuracy loss",
      "Increases the embedding dimension", "Removes the need for a model",
      "Guarantees perfect recall"),
    q("Caching the model with st.cache_resource avoids:",
      "Reloading the transformer on every rerun",
      "Displaying the results", "Computing cosine similarity",
      "Reading the CSV file"),
    q("In Streamlit, session state is used here to keep:",
      "The index, results and trials across reruns",
      "The CSS stylesheet", "The model weights on disk",
      "The user's password"),
    q("Storing the dataset in data/documents.csv rather than in code makes the system:",
      "Easier to update without editing the application",
      "Impossible to extend", "Faster to encode", "Independent of the model"),
    q("When a user uploads documents, the index signature changes so that:",
      "The application knows the stored index is stale",
      "The uploaded file is deleted", "The threshold resets to 1",
      "The model is retrained"),
    q("Which metric best shows how confidently the top result beats the rest?",
      "The gap between the first and second similarity scores",
      "The total number of documents", "The index build time",
      "The number of categories"),
    q("If every score in a search falls below 0.2, the most likely reason is:",
      "The collection contains nothing related to the query",
      "The embedding dimension is wrong", "Top-K is too large",
      "The chart failed to render"),
    q("A search over a 100,000 document collection would most likely need:",
      "An approximate nearest neighbour index",
      "A larger similarity threshold only", "Fewer categories",
      "A smaller Top-K only"),
    q("The main computational cost at query time in this system is:",
      "Encoding the query and scanning the index",
      "Rebuilding all document embeddings", "Reloading the CSS",
      "Regenerating the PDF report"),
    q("Normalising embeddings at encode time lets similarity be computed as:",
      "A single matrix multiplication of unit vectors",
      "A sort operation", "A string comparison", "A hash lookup"),
    q("Recording trials during the experiment is useful because it:",
      "Lets you compare how parameters change the retrieved results",
      "Speeds up the encoder", "Reduces the index size",
      "Removes irrelevant documents permanently"),
    q("Which change would most improve recall for a paraphrased query?",
      "Lowering the similarity threshold", "Raising the threshold",
      "Reducing Top-K to 1", "Removing the index"),
    q("Semantic search applied to customer support mainly helps by:",
      "Matching a user's wording to differently worded help articles",
      "Encrypting the tickets", "Deleting old tickets",
      "Assigning staff schedules"),
    q("In plagiarism detection, dense embeddings help detect:",
      "Reworded passages that share meaning",
      "Only exact copied sentences", "Font and layout changes",
      "File compression artefacts"),
    q("A practical privacy concern with embedding services is that:",
      "Document text may leave the organisation during encoding",
      "Vectors cannot be stored", "Cosine similarity is reversible by design",
      "Indexes cannot be deleted"),
    q("The overall conclusion of this experiment is that:",
      "Dense embeddings with cosine similarity retrieve semantically relevant documents",
      "Keyword matching is always superior", "Embeddings cannot be compared numerically",
      "Ranking requires no scoring function"),
]

# ============================================================
# CONCEPT ANIMATION (mock data, purely illustrative)
# ============================================================
#
# This is a self-contained HTML/CSS/JS widget rendered through
# components.html(). It uses invented documents and coordinates --
# no real embeddings -- purely to give a visual, animated picture of
# what the backend is doing: documents become points in a space,
# a query becomes a point too, and the nearest points are the answer.

CONCEPT_ANIMATION_HTML = r"""
<div id="root">
  <div class="toolbar">
    <button id="playBtn">&#9654; Play Animation</button>
    <button id="replayBtn">&#8635; Replay</button>
    <label class="speedLabel">Speed:
      <select id="speedSelect">
        <option value="0.5">0.5x (Slow)</option>
        <option value="1" selected>1x (Normal)</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x (Fast)</option>
        <option value="3">3x (Very Fast)</option>
      </select>
    </label>
    <div id="caption">Click Play to see how semantic search works, step by step.</div>
  </div>

  <div class="legend">
    <span class="chip"><i style="background:#4f8ff0"></i> AI &amp; ML topics</span>
    <span class="chip"><i style="background:#37b06a"></i> Systems topics</span>
    <span class="chip"><i style="background:#b06fe0"></i> Data topics</span>
    <span class="chip"><i style="background:#f47721;border-radius:50%"></i> Your query</span>
  </div>

  <div id="stepFlow" class="stepFlow"></div>

  <svg id="stage" viewBox="0 0 960 460" width="100%" height="430" preserveAspectRatio="xMidYMid meet">
    <text x="95" y="26" class="panelLabel">Documents</text>
    <rect x="15" y="36" width="165" height="380" rx="10" class="panel"/>
    <g id="docStack"></g>

    <polygon points="188,225 214,215 214,235" class="flowArrow" id="arrow1"/>
    <text x="201" y="255" class="arrowLabel">reads</text>

    <text x="425" y="26" class="panelLabel">Encoder Model</text>
    <rect id="encoderBox" x="345" y="185" width="150" height="90" rx="12" class="encoder"/>
    <text x="420" y="224" class="encoderText">Sentence</text>
    <text x="420" y="242" class="encoderText">Transformer</text>

    <polygon points="503,225 529,215 529,235" class="flowArrow" id="arrow2"/>
    <text x="516" y="255" class="arrowLabel">embeds</text>

    <text x="775" y="26" class="panelLabel">Embedding Space</text>
    <rect x="590" y="36" width="355" height="380" rx="10" class="panel"/>
    <g id="dotsLayer"></g>
    <g id="linesLayer"></g>
    <g id="travelLayer"></g>
    <g id="queryLayer"></g>
    <g id="rankLayer"></g>
  </svg>

  <div id="resultList" class="resultList"></div>
</div>

<style>
#root {
    font-family: -apple-system, "Segoe UI", Arial, sans-serif;
    background: #ffffff;
}

.toolbar {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

.toolbar button {
    background: #2696d2;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 9px 18px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
}

.toolbar button:hover { background: #197db6; }

.speedLabel {
    font-size: 13px;
    color: #555555;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.speedLabel select {
    font-size: 13px;
    padding: 5px 8px;
    border: 1px solid #cccccc;
    border-radius: 4px;
    background: white;
    color: #333333;
}

#caption {
    color: #444444;
    font-size: 15px;
    font-weight: 500;
    min-height: 20px;
}

.legend {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    margin-bottom: 6px;
}

.stepFlow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin: 4px 0px 14px 0px;
}

.stepPill {
    background: #f7f7f7;
    border: 1px solid #e2e2e2;
    color: #9a9a9a;
    border-radius: 14px;
    padding: 6px 13px;
    font-size: 12.5px;
    font-weight: 600;
    white-space: nowrap;
    transition: background 300ms ease, border-color 300ms ease, color 300ms ease;
}

.stepPill.stepDone {
    background: #eaf7ee;
    border-color: #9bd5ad;
    color: #1d6b38;
}

.stepPill.stepActive {
    background: #f47721;
    border-color: #f47721;
    color: #ffffff;
    box-shadow: 0 0 0 3px rgba(244, 119, 33, 0.18);
}

.stepArrow {
    color: #c7c7c7;
    font-size: 14px;
}

.flowArrow {
    fill: #d7d7d7;
    transition: fill 300ms ease;
}

.flowArrow.flowing {
    fill: #f47721;
}

.arrowLabel {
    font-size: 10px;
    fill: #999999;
    text-anchor: middle;
}

.travelDot {
    filter: drop-shadow(0 0 2px rgba(0,0,0,0.25));
}

.chip {
    font-size: 13px;
    color: #555555;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.chip i {
    width: 10px;
    height: 10px;
    display: inline-block;
    border-radius: 2px;
}

.panel {
    fill: #f7fafc;
    stroke: #dbe6ee;
    stroke-width: 1;
}

.panelLabel {
    font-size: 15px;
    fill: #2d9bd3;
    font-weight: 600;
    text-anchor: middle;
}

.encoder {
    fill: #eaf3fb;
    stroke: #2696d2;
    stroke-width: 1.5;
    transform-box: fill-box;
    transform-origin: center;
}

.encoder.pulse {
    animation: pulseKey 480ms ease;
}

@keyframes pulseKey {
    0%   { transform: scale(1); }
    45%  { transform: scale(1.1); filter: drop-shadow(0 0 6px #2696d2aa); }
    100% { transform: scale(1); }
}

.encoderText {
    font-size: 13px;
    fill: #23648c;
    text-anchor: middle;
    font-weight: 600;
}

.docCard {
    transition: opacity 250ms ease;
}

.docCard rect {
    stroke-width: 1.4;
    fill: #ffffff;
}

.docCard.active rect {
    stroke: #f47721;
    stroke-width: 2.4;
}

.docCard.done {
    opacity: 0.35;
}

.docLabel {
    font-size: 9px;
    fill: #555555;
}

.dot {
    transform-box: fill-box;
    transform-origin: center;
    transform: scale(0);
    opacity: 0;
    transition: transform 480ms cubic-bezier(.34,1.56,.64,1), opacity 300ms ease;
}

.dot.shown {
    transform: scale(1);
    opacity: 1;
}

.queryChip {
    opacity: 0;
    transition: opacity 350ms ease;
}

.queryChip.shown { opacity: 1; }

.queryDot {
    transform-box: fill-box;
    transform-origin: center;
    transform: scale(0);
    opacity: 0;
    transition: transform 500ms cubic-bezier(.34,1.56,.64,1), opacity 300ms ease;
}

.queryDot.shown {
    transform: scale(1);
    opacity: 1;
}

.simLine {
    stroke: #b9c4cc;
    stroke-width: 1;
    opacity: 0;
    transition: opacity 500ms ease, stroke 400ms ease, stroke-width 400ms ease;
}

.simLine.shown { opacity: 0.45; }
.simLine.hot { stroke: #f47721; stroke-width: 2.6; opacity: 0.9; }
.simLine.cold { opacity: 0.08; }

.rankBadge {
    transform-box: fill-box;
    transform-origin: center;
    transform: scale(0);
    opacity: 0;
    transition: transform 420ms cubic-bezier(.34,1.56,.64,1), opacity 300ms ease;
}

.rankBadge.shown { transform: scale(1); opacity: 1; }

.rankBadge circle { fill: #f47721; }
.rankBadge text { fill: white; font-size: 12px; font-weight: 700; text-anchor: middle; }

.resultList {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.resultRow {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #ffffff;
    border: 1px solid #e2e2e2;
    border-left: 4px solid #f47721;
    border-radius: 5px;
    padding: 8px 12px;
    opacity: 0;
    transform: translateY(10px);
    transition: opacity 400ms ease, transform 400ms ease;
}

.resultRow.shown { opacity: 1; transform: translateY(0); }

.resultRank {
    font-weight: 700;
    color: #f47721;
    width: 22px;
}

.resultTitle {
    flex: 1;
    font-size: 14px;
    color: #333333;
}

.resultBarTrack {
    width: 120px;
    height: 8px;
    background: #eeeeee;
    border-radius: 4px;
    overflow: hidden;
}

.resultBarFill {
    height: 100%;
    width: 0%;
    background: #198754;
    transition: width 700ms ease;
}

.resultScore {
    width: 46px;
    text-align: right;
    font-size: 13px;
    color: #198754;
    font-weight: 600;
}
</style>

<script>
(function () {
    const svgNS = "http://www.w3.org/2000/svg";
    const root = document.getElementById("root");
    const docStack = root.querySelector("#docStack");
    const dotsLayer = root.querySelector("#dotsLayer");
    const linesLayer = root.querySelector("#linesLayer");
    const travelLayer = root.querySelector("#travelLayer");
    const queryLayer = root.querySelector("#queryLayer");
    const rankLayer = root.querySelector("#rankLayer");
    const encoderBox = root.querySelector("#encoderBox");
    const caption = root.querySelector("#caption");
    const resultList = root.querySelector("#resultList");
    const stepFlowEl = root.querySelector("#stepFlow");
    const arrow1 = root.querySelector("#arrow1");
    const arrow2 = root.querySelector("#arrow2");

    const STEPS = [
        "1. Read Documents",
        "2. Encode",
        "3. Build Index",
        "4. Read Query",
        "5. Compare & Rank",
        "6. Show Results",
    ];

    const COLORS = { ai: "#4f8ff0", sys: "#37b06a", data: "#b06fe0" };

    // Mock documents: a stack icon on the left and a target point
    // in the embedding space on the right, grouped into three
    // loose visual clusters purely for illustration.
    const DOCS = [
        { title: "Intro to Machine Learning", cluster: "ai", tx: 660, ty: 110 },
        { title: "Neural Networks Basics",     cluster: "ai", tx: 700, ty: 90  },
        { title: "Supervised Learning",        cluster: "ai", tx: 675, ty: 150 },
        { title: "Deep Learning Overview",     cluster: "ai", tx: 720, ty: 130 },
        { title: "NLP Fundamentals",           cluster: "ai", tx: 650, ty: 160 },
        { title: "Operating Systems",          cluster: "sys", tx: 860, ty: 240 },
        { title: "Computer Networks",          cluster: "sys", tx: 900, ty: 260 },
        { title: "Process Scheduling",         cluster: "sys", tx: 870, ty: 290 },
        { title: "Memory Management",          cluster: "sys", tx: 910, ty: 220 },
        { title: "Database Systems",           cluster: "data", tx: 720, ty: 360 },
        { title: "Data Warehousing",           cluster: "data", tx: 760, ty: 390 },
        { title: "Vector Databases",           cluster: "data", tx: 690, ty: 380 },
    ];

    const QUERY_TEXT = "How do machines learn from data?";
    const QUERY_TARGET = { tx: 690, ty: 130 };
    const TOP_K = [0, 3, 4]; // indices into DOCS considered "most similar"
    const MOCK_SCORES = { 0: 0.91, 3: 0.85, 4: 0.79 };

    function speedFactor() {
        const sel = document.getElementById("speedSelect");
        return sel ? parseFloat(sel.value) : 1;
    }

    function scaledMs(base) { return Math.max(60, base / speedFactor()); }

    function sleep(ms) { return new Promise(r => setTimeout(r, ms / speedFactor())); }

    function setCaption(text) { caption.textContent = text; }

    function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

    const DOC_START_Y = 55;
    const DOC_GAP = 30;

    function buildStepFlow() {
        clear(stepFlowEl);
        STEPS.forEach((label, i) => {
            if (i > 0) {
                const arrow = document.createElement("span");
                arrow.className = "stepArrow";
                arrow.innerHTML = "&rarr;";
                stepFlowEl.appendChild(arrow);
            }
            const pill = document.createElement("span");
            pill.className = "stepPill";
            pill.id = "step-" + i;
            pill.textContent = label;
            stepFlowEl.appendChild(pill);
        });
    }

    function setStep(activeIndex) {
        STEPS.forEach((label, i) => {
            const pill = document.getElementById("step-" + i);
            pill.classList.remove("stepDone", "stepActive");
            if (i < activeIndex) pill.classList.add("stepDone");
            else if (i === activeIndex) pill.classList.add("stepActive");
        });
    }

    // Interpolates numeric SVG attributes on el from fromVals to toVals over
    // duration ms, using requestAnimationFrame so it works the same across
    // browsers regardless of speed setting. Resolves when the motion ends.
    function tween(el, fromVals, toVals, duration) {
        return new Promise(resolve => {
            const t0 = performance.now();

            function frame(now) {
                const t = Math.min(1, (now - t0) / duration);
                const eased = 1 - Math.pow(1 - t, 3);

                for (const key in toVals) {
                    const v = fromVals[key] + (toVals[key] - fromVals[key]) * eased;
                    el.setAttribute(key, v);
                }

                if (t < 1) {
                    requestAnimationFrame(frame);
                } else {
                    resolve();
                }
            }

            requestAnimationFrame(frame);
        });
    }

    function buildDocStack() {
        clear(docStack);

        DOCS.forEach((doc, i) => {
            const g = document.createElementNS(svgNS, "g");
            g.setAttribute("class", "docCard");
            g.setAttribute("id", "doc-" + i);

            const rect = document.createElementNS(svgNS, "rect");
            rect.setAttribute("x", 30);
            rect.setAttribute("y", DOC_START_Y + i * DOC_GAP);
            rect.setAttribute("width", 130);
            rect.setAttribute("height", 20);
            rect.setAttribute("rx", 4);
            rect.setAttribute("stroke", COLORS[doc.cluster]);
            g.appendChild(rect);

            const label = document.createElementNS(svgNS, "text");
            label.setAttribute("x", 36);
            label.setAttribute("y", DOC_START_Y + i * DOC_GAP + 14);
            label.setAttribute("class", "docLabel");
            label.textContent = doc.title.length > 22 ? doc.title.slice(0, 20) + "..." : doc.title;
            g.appendChild(label);

            docStack.appendChild(g);
        });
    }

    function makeDot(x, y, color, extraClass) {
        const c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", x);
        c.setAttribute("cy", y);
        c.setAttribute("r", 7);
        c.setAttribute("fill", color);
        c.setAttribute("class", "dot" + (extraClass ? " " + extraClass : ""));
        c.style.transition =
            "transform " + scaledMs(480) + "ms cubic-bezier(.34,1.56,.64,1), " +
            "opacity " + scaledMs(300) + "ms ease";
        return c;
    }

    function makeTravelDot(x, y, color) {
        const c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", x);
        c.setAttribute("cy", y);
        c.setAttribute("r", 6);
        c.setAttribute("fill", color);
        c.setAttribute("class", "travelDot");
        return c;
    }

    // Moves a small dot from the document's row, through the encoder, out to
    // its final landing point -- a literal, visible flow between the three
    // panels instead of an instant appear/disappear.
    async function flowDocumentToEmbedding(doc, index) {
        const rowY = DOC_START_Y + index * DOC_GAP + 10;
        const color = COLORS[doc.cluster];

        arrow1.classList.add("flowing");

        const travelDot = makeTravelDot(170, rowY, color);
        travelLayer.appendChild(travelDot);

        await tween(
            travelDot,
            { cx: 170, cy: rowY },
            { cx: 420, cy: 230 },
            scaledMs(360)
        );

        arrow1.classList.remove("flowing");
        await pulseEncoder();
        arrow2.classList.add("flowing");

        await tween(
            travelDot,
            { cx: 420, cy: 230 },
            { cx: doc.tx, cy: doc.ty },
            scaledMs(360)
        );

        arrow2.classList.remove("flowing");
        travelDot.remove();

        const dot = makeDot(doc.tx, doc.ty, color);
        dotsLayer.appendChild(dot);
        requestAnimationFrame(() => dot.classList.add("shown"));
    }

    function reset() {
        buildDocStack();
        buildStepFlow();
        clear(dotsLayer);
        clear(linesLayer);
        clear(travelLayer);
        clear(queryLayer);
        clear(rankLayer);
        clear(resultList);
        arrow1.classList.remove("flowing");
        arrow2.classList.remove("flowing");
        setCaption("Click Play to see how semantic search works, step by step.");
    }

    async function pulseEncoder() {
        encoderBox.style.animationDuration = scaledMs(480) + "ms";
        encoderBox.classList.add("pulse");
        await sleep(480);
        encoderBox.classList.remove("pulse");
    }

    async function playAnimation() {
        reset();
        await sleep(300);

        // ---- Phase 1: index every document ----
        setStep(0);
        setCaption("Step 1 of 6 — Every document is read by the model, one at a time.");
        await sleep(400);

        for (let i = 0; i < DOCS.length; i++) {
            const doc = DOCS[i];
            const card = document.getElementById("doc-" + i);
            card.classList.add("active");

            if (i === Math.floor(DOCS.length / 2)) {
                setStep(1);
                setCaption("Step 2 of 6 — The encoder converts each document into a vector.");
            }

            await flowDocumentToEmbedding(doc, i);

            card.classList.remove("active");
            card.classList.add("done");

            await sleep(120);
        }

        setStep(2);
        setCaption("Step 3 of 6 — All documents are now points in the embedding index. Similar topics land close together.");
        await sleep(1300);

        // ---- Phase 2: embed the query ----
        setStep(3);
        setCaption("Step 4 of 6 — A new query from the user is typed in.");

        const chip = document.createElementNS(svgNS, "g");
        chip.setAttribute("class", "queryChip");
        chip.style.transition = "opacity " + scaledMs(350) + "ms ease";
        const chipRect = document.createElementNS(svgNS, "rect");
        chipRect.setAttribute("x", 30);
        chipRect.setAttribute("y", 400);
        chipRect.setAttribute("width", 145);
        chipRect.setAttribute("height", 24);
        chipRect.setAttribute("rx", 12);
        chipRect.setAttribute("fill", "#fff1e4");
        chipRect.setAttribute("stroke", "#f47721");
        chip.appendChild(chipRect);
        const chipText = document.createElementNS(svgNS, "text");
        chipText.setAttribute("x", 40);
        chipText.setAttribute("y", 416);
        chipText.setAttribute("style", "font-size:9px;fill:#b4530c;");
        chipText.textContent = "\"" + QUERY_TEXT.slice(0, 26) + "...\"";
        chip.appendChild(chipText);
        queryLayer.appendChild(chip);
        requestAnimationFrame(() => chip.classList.add("shown"));

        await sleep(700);
        setCaption("The query is passed through the same encoder model as the documents.");

        arrow1.classList.add("flowing");
        const queryTravelDot = makeTravelDot(170, 412, "#f47721");
        travelLayer.appendChild(queryTravelDot);

        await tween(
            queryTravelDot,
            { cx: 170, cy: 412 },
            { cx: 420, cy: 230 },
            scaledMs(400)
        );

        arrow1.classList.remove("flowing");
        chip.classList.remove("shown");
        await pulseEncoder();
        arrow2.classList.add("flowing");

        await tween(
            queryTravelDot,
            { cx: 420, cy: 230 },
            { cx: QUERY_TARGET.tx, cy: QUERY_TARGET.ty },
            scaledMs(400)
        );

        arrow2.classList.remove("flowing");
        queryTravelDot.remove();

        const queryDot = document.createElementNS(svgNS, "circle");
        queryDot.setAttribute("cx", QUERY_TARGET.tx);
        queryDot.setAttribute("cy", QUERY_TARGET.ty);
        queryDot.setAttribute("r", 9);
        queryDot.setAttribute("fill", "#f47721");
        queryDot.setAttribute("class", "queryDot");
        queryDot.style.transition =
            "transform " + scaledMs(500) + "ms cubic-bezier(.34,1.56,.64,1), " +
            "opacity " + scaledMs(300) + "ms ease";
        queryLayer.appendChild(queryDot);
        requestAnimationFrame(() => queryDot.classList.add("shown"));

        await sleep(500);

        // ---- Phase 3: compare with every document ----
        setStep(4);
        setCaption("Step 5 of 6 — The query's position is compared with every document using cosine similarity.");

        DOCS.forEach((doc, i) => {
            const line = document.createElementNS(svgNS, "line");
            line.setAttribute("x1", QUERY_TARGET.tx);
            line.setAttribute("y1", QUERY_TARGET.ty);
            line.setAttribute("x2", doc.tx);
            line.setAttribute("y2", doc.ty);
            line.setAttribute("class", "simLine");
            line.setAttribute("id", "line-" + i);
            line.style.transition =
                "opacity " + scaledMs(500) + "ms ease, " +
                "stroke " + scaledMs(400) + "ms ease, " +
                "stroke-width " + scaledMs(400) + "ms ease";
            linesLayer.appendChild(line);
            requestAnimationFrame(() => line.classList.add("shown"));
        });

        await sleep(1100);

        // ---- Phase 4: rank and highlight the closest ----
        setCaption("Ranking by similarity — the closest points are the most relevant documents.");

        DOCS.forEach((doc, i) => {
            const line = document.getElementById("line-" + i);
            if (TOP_K.includes(i)) {
                line.classList.add("hot");
            } else {
                line.classList.add("cold");
            }
        });

        await sleep(500);

        TOP_K.forEach((docIndex, rank) => {
            const doc = DOCS[docIndex];
            const badge = document.createElementNS(svgNS, "g");
            badge.setAttribute("class", "rankBadge");
            badge.style.transition =
                "transform " + scaledMs(420) + "ms cubic-bezier(.34,1.56,.64,1), " +
                "opacity " + scaledMs(300) + "ms ease";
            const circle = document.createElementNS(svgNS, "circle");
            circle.setAttribute("cx", doc.tx + 14);
            circle.setAttribute("cy", doc.ty - 14);
            circle.setAttribute("r", 10);
            badge.appendChild(circle);
            const text = document.createElementNS(svgNS, "text");
            text.setAttribute("x", doc.tx + 14);
            text.setAttribute("y", doc.ty - 10);
            text.textContent = String(rank + 1);
            badge.appendChild(text);
            rankLayer.appendChild(badge);
            requestAnimationFrame(() => badge.classList.add("shown"));
        });

        await sleep(700);
        setCaption("Ranked results are returned to the user, best match first.");
        setStep(5);

        TOP_K.forEach((docIndex, rank) => {
            const doc = DOCS[docIndex];
            const score = MOCK_SCORES[docIndex];

            const row = document.createElement("div");
            row.className = "resultRow";
            row.style.transition =
                "opacity " + scaledMs(400) + "ms ease, transform " + scaledMs(400) + "ms ease";

            const rankEl = document.createElement("div");
            rankEl.className = "resultRank";
            rankEl.textContent = "#" + (rank + 1);

            const titleEl = document.createElement("div");
            titleEl.className = "resultTitle";
            titleEl.textContent = doc.title;

            const trackEl = document.createElement("div");
            trackEl.className = "resultBarTrack";
            const fillEl = document.createElement("div");
            fillEl.className = "resultBarFill";
            fillEl.style.transition = "width " + scaledMs(700) + "ms ease";
            trackEl.appendChild(fillEl);

            const scoreEl = document.createElement("div");
            scoreEl.className = "resultScore";
            scoreEl.textContent = score.toFixed(2);

            row.appendChild(rankEl);
            row.appendChild(titleEl);
            row.appendChild(trackEl);
            row.appendChild(scoreEl);
            resultList.appendChild(row);

            requestAnimationFrame(() => {
                row.classList.add("shown");
                setTimeout(() => { fillEl.style.width = (score * 100) + "%"; }, 120);
            });
        });
    }

    document.getElementById("playBtn").addEventListener("click", () => { playAnimation(); });
    document.getElementById("replayBtn").addEventListener("click", () => { reset(); });

    reset();
})();
</script>
"""


def render_concept_animation():
    render_html('<div class="content-subheading">How Semantic Search Works — Animated Overview</div>')

    st.caption(
        "This walkthrough uses made-up documents and positions purely to illustrate the "
        "idea. The real dataset and your own queries are used further down, in "
        "\"Try It on the Real Dataset\"."
    )

    components.html(CONCEPT_ANIMATION_HTML, height=690, scrolling=False)

# ============================================================
# EXTRA STYLES FOR THE LIVE PROCESS VIEW
# ============================================================

render_html(
    """
    <style>

    .pipeline-stage.stage-done {
        background: #eaf7ee;
        border-color: #9bd5ad;
        color: #1d6b38;
    }

    .pipeline-stage.stage-active {
        background: #fff1e4;
        border-color: #f4a26a;
        color: #b4530c;
        box-shadow: 0 0 0 2px rgba(244, 119, 33, 0.18);
    }

    .pipeline-stage.stage-todo {
        background: #f7f7f7;
        border-color: #e2e2e2;
        color: #9a9a9a;
    }

    .live-log {
        background: #0f1b24;
        color: #d7f0ff;
        font-family: "Consolas", "Monaco", monospace;
        font-size: 13px;
        line-height: 1.55;
        padding: 14px 16px;
        border-radius: 5px;
        min-height: 120px;
        white-space: pre-wrap;
    }

    .live-log .log-ok { color: #7ee2a8; }
    .live-log .log-run { color: #ffc166; }

    .quiz-progress {
        color: #666666;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .app-card {
        border: 1px solid #e2e2e2;
        border-top: 4px solid #2696d2;
        border-radius: 5px;
        padding: 16px 18px;
        margin: 10px 0px;
        background: #ffffff;
        height: 100%;
    }

    .app-card-title {
        color: #2696d2;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .app-card-text {
        color: #333333;
        font-size: 15px;
        line-height: 1.6;
    }

    </style>
    """
)


# ============================================================
# DATASET CONSTRUCTION AND LOADING
# ============================================================

REQUIRED_COLUMNS = ["id", "title", "category", "content"]


def build_default_documents():
    """Expand CORPUS into a full DataFrame with ids, source and word count."""
    rows = []
    counter = 1

    for category, entries in CORPUS.items():
        for position, (title, content) in enumerate(entries):
            rows.append(
                {
                    "id": f"DOC{counter:03d}",
                    "title": title,
                    "category": category,
                    "content": content,
                    "source": SOURCE_TYPES[position % len(SOURCE_TYPES)],
                    "word_count": len(content.split()),
                }
            )
            counter += 1

    return pd.DataFrame(rows)


def normalize_documents(df, id_prefix="DOC"):
    """Make any incoming frame conform to the expected schema."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "document id": "id",
        "doc_id": "id",
        "docid": "id",
        "text": "content",
        "body": "content",
        "document": "content",
        "topic": "category",
        "label": "category",
        "name": "title",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "content" not in df.columns:
        raise ValueError("The dataset must contain a 'content' column.")

    if "title" not in df.columns:
        df["title"] = df["content"].astype(str).str.slice(0, 60) + "..."

    if "category" not in df.columns:
        df["category"] = "Uncategorized"

    if "source" not in df.columns:
        df["source"] = "User Upload"

    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"].str.len() > 0]

    if "id" not in df.columns:
        df["id"] = [f"{id_prefix}{i:03d}" for i in range(1, len(df) + 1)]

    df["word_count"] = df["content"].astype(str).str.split().str.len()

    return df[
        ["id", "title", "category", "content", "source", "word_count"]
    ].reset_index(drop=True)


@st.cache_data
def load_base_documents():
    """Prefer data/documents.csv; fall back to the built-in corpus."""
    if DATA_FILE.exists():
        try:
            return normalize_documents(pd.read_csv(DATA_FILE)), "data/documents.csv"
        except Exception:
            pass

    return build_default_documents(), "Built-in corpus"


def parse_uploaded_file(uploaded_file):
    """Turn an uploaded csv / json / txt file into a normalised frame."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(raw))

    elif name.endswith(".json"):
        payload = json.loads(raw.decode("utf-8", "replace"))
        if isinstance(payload, dict):
            for key in ("documents", "data", "records"):
                if key in payload:
                    payload = payload[key]
                    break
        frame = pd.DataFrame(payload)

    elif name.endswith(".txt"):
        text = raw.decode("utf-8", "replace")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if not blocks:
            blocks = [line.strip() for line in text.splitlines() if line.strip()]

        frame = pd.DataFrame(
            {
                "title": [b.split("\n")[0][:70] for b in blocks],
                "category": ["Uploaded Text"] * len(blocks),
                "content": [" ".join(b.split()) for b in blocks],
            }
        )

    else:
        raise ValueError("Only CSV, JSON and TXT files are supported.")

    return normalize_documents(frame, id_prefix="UPL")


def get_active_documents():
    """Default collection plus anything the user has uploaded."""
    base_df, _ = load_base_documents()

    uploaded = st.session_state.get("uploaded_documents")

    if uploaded is not None and len(uploaded) > 0:
        return pd.concat([base_df, uploaded], ignore_index=True)

    return base_df


def dataset_signature(df):
    """Cheap fingerprint used to detect that the index is out of date."""
    return f"{len(df)}:{hash(tuple(df['id'].tolist()))}"


# ============================================================
# THEORY TEXT
# ============================================================

THEORY_AIM = """
To implement a semantic search system using dense sentence embeddings
and cosine similarity for retrieving documents that are semantically
related to a user's query.
"""

THEORY_OBJECTIVES = [
    "Understand dense text embeddings and the vector space they live in.",
    "Generate vector representations using SentenceTransformer.",
    "Build and inspect an embedding index over a document collection.",
    "Understand cosine similarity between vectors.",
    "Retrieve semantically similar documents for a natural language query.",
    "Analyze the effect of Top-K and of the minimum similarity threshold.",
]

THEORY_INTRODUCTION = """
Traditional keyword-based search mainly matches exact words between a
query and a document. It may fail when two texts have similar meanings
but use different words.

Dense embedding-based semantic search represents text as numerical
vectors. These vectors capture semantic meaning using a pretrained
transformer model.

During search, the query is converted into an embedding. The embedding
is compared with document embeddings using cosine similarity. Documents
with higher similarity scores are considered more relevant to the query.
"""

THEORY_EMBEDDINGS = """
A dense embedding is a fixed-length numerical vector representing the
meaning of a text.

For example, the query:

How do machines learn from data?

may be semantically related to a document about machine learning even
if both texts do not contain exactly the same words.
"""

THEORY_COSINE = """
Cosine similarity measures the angle between two vectors.

A value close to 1 indicates high similarity.
A value close to 0 indicates low similarity.

The formula is:

cosine_similarity(A, B) =
(A dot B) / (||A|| multiplied by ||B||)
"""

PROCEDURE_STEPS = [
    "Load the document collection from the dataset file.",
    "Optionally upload an additional document collection.",
    "Load the pretrained SentenceTransformer model.",
    "Build the embedding index for all documents and watch Stage A run live.",
    "Inspect the generated embeddings, the index statistics and the embedding map.",
    "Enter a natural language query (Stage B).",
    "Follow the live query pipeline: preprocessing, encoding, similarity, ranking.",
    "Apply the similarity threshold and keep the Top-K documents.",
    "Record the trial and repeat with different parameters.",
    "Generate the PDF report.",
]

APPLICATION_AREAS = [
    (
        "Web and Enterprise Search",
        "Search engines combine a lexical index with dense retrieval so that a query "
        "such as 'how do I stop my laptop from overheating' matches a document titled "
        "'thermal throttling and fan maintenance', even though almost no words overlap. "
        "In enterprise search the same approach lets employees find internal policies, "
        "wikis and reports written in vocabulary they do not know in advance.",
    ),
    (
        "Retrieval Augmented Generation",
        "A large language model cannot know private or very recent data. In retrieval "
        "augmented generation the user question is embedded, the most similar document "
        "chunks are retrieved from a vector index, and those chunks are placed in the "
        "model's prompt as context. This grounds the answer in real sources, reduces "
        "hallucination and makes citation possible.",
    ),
    (
        "Question Answering and Chatbots",
        "Support chatbots embed the incoming question and retrieve the closest entries "
        "from a knowledge base of previously answered questions. Because matching is "
        "semantic, 'my card was declined' can be routed to an article titled 'payment "
        "authorisation failures' without any keyword rule being written by hand.",
    ),
    (
        "Customer Support Ticket Routing",
        "Incoming tickets are embedded and compared with historical tickets and their "
        "resolutions. Similar past cases are shown to the agent, and clusters in the "
        "embedding space reveal recurring problems that deserve a permanent fix rather "
        "than repeated manual handling.",
    ),
    (
        "Recommendation Systems",
        "Articles, courses, products or videos are embedded from their descriptions, and "
        "an item is recommended when its vector is close to the vectors of items the user "
        "already engaged with. This content based signal works for new items that have no "
        "interaction history, which is where collaborative filtering fails.",
    ),
    (
        "Research Paper and Patent Search",
        "Researchers rarely know the exact terminology used by every related field. "
        "Embedding abstracts allows a search for 'predicting protein folding with neural "
        "networks' to surface relevant work that uses entirely different wording, and "
        "nearest neighbour search over the same index reveals prior art for patents.",
    ),
    (
        "Legal and Regulatory Document Retrieval",
        "Contracts, judgements and regulations are long and highly repetitive. Semantic "
        "retrieval over clause level chunks lets a lawyer find every clause that limits "
        "liability, however it is phrased, and compare it against a standard template.",
    ),
    (
        "Medical and Clinical Information Retrieval",
        "Clinical notes mix abbreviations, drug names and informal descriptions. Dense "
        "retrieval links a symptom description to relevant guidelines and literature, "
        "supporting decision making. Such systems are used as assistive tools and their "
        "output is always reviewed by a qualified clinician.",
    ),
    (
        "Plagiarism and Duplicate Detection",
        "Exact string matching misses paraphrased text. Because embeddings encode meaning, "
        "two passages that say the same thing in different words still land close together, "
        "which exposes reworded copying and near duplicate records in a dataset.",
    ),
    (
        "E-commerce Product Discovery",
        "Shoppers describe what they want rather than what a product is called. Embedding "
        "product titles and descriptions lets 'shoes for standing all day at work' retrieve "
        "cushioned work footwear, and the same vectors power 'similar items' carousels.",
    ),
    (
        "Educational Platforms and Virtual Labs",
        "Learning platforms embed lessons, exercises and past questions so a student query "
        "returns the exact section that explains the concept. The same technique can map a "
        "student's wrong answer to the topic they should revise.",
    ),
    (
        "Multilingual and Cross-lingual Search",
        "Multilingual encoders place a sentence and its translation near each other in one "
        "shared space, so a query written in Hindi or Marathi can retrieve relevant English "
        "documents without any machine translation step at query time.",
    ),
]

APPLICATION_LIMITATIONS = [
    "Exact identifiers such as product codes, error numbers or citations are better handled by lexical matching, so production systems usually combine both.",
    "An encoder performs worse on domains very different from its training data, and may need fine tuning on in-domain text.",
    "Long documents must be split into chunks, because one vector cannot faithfully represent many pages.",
    "Brute force scanning becomes slow on very large collections, so approximate nearest neighbour indexes such as HNSW or IVF are used.",
    "Embedding text with a hosted service sends that text outside the organisation, which raises privacy and compliance questions.",
]


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    defaults = {
        "active_section": "Aim and Introduction",
        "last_query": "",
        "last_results": [],
        "last_query_time": 0.0,
        "last_embedding_dimension": 0,
        "last_documents_compared": 0,
        "last_threshold": 0.25,
        "last_score_distribution": None,
        "trials": [],
        "index_built": False,
        "document_embeddings": None,
        "indexed_documents": None,
        "index_signature": "",
        "index_build_time": 0.0,
        "uploaded_documents": None,
        "upload_message": "",
        "pretest_submitted": False,
        "pretest_score": 0,
        "pretest_attempted": 0,
        "pretest_selected": None,
        "posttest_submitted": False,
        "posttest_score": 0,
        "posttest_attempted": 0,
        "posttest_selected": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# MODEL, INDEXING AND SEARCH
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def pca_projection(embeddings, components=2):
    """Small numpy PCA used only to visualise the embedding space."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:components].T


def store_index(embeddings, documents_df, elapsed):
    st.session_state.document_embeddings = embeddings
    st.session_state.indexed_documents = documents_df.reset_index(drop=True)
    st.session_state.index_signature = dataset_signature(documents_df)
    st.session_state.index_build_time = elapsed
    st.session_state.index_built = True


def index_is_current(documents_df):
    return (
        st.session_state.index_built
        and st.session_state.document_embeddings is not None
        and st.session_state.index_signature == dataset_signature(documents_df)
    )


def compute_similarities(query, documents_df, embeddings):
    model = load_embedding_model()

    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embeddings)[0]

    return query_embedding[0], scores


def assemble_results(scores, documents_df, top_k, threshold):
    order = np.argsort(scores)[::-1]

    results = []

    for index in order:
        score = float(scores[index])

        if score < threshold:
            continue

        row = documents_df.iloc[index]

        results.append(
            {
                "Rank": len(results) + 1,
                "Document ID": row["id"],
                "Title": row["title"],
                "Category": row["category"],
                "Source": row.get("source", "-"),
                "Word Count": int(row.get("word_count", 0)),
                "Content": row["content"],
                "Similarity": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


def record_trial():
    results = st.session_state.last_results

    if not results:
        return

    best = results[0]

    st.session_state.trials.append(
        {
            "Trial": len(st.session_state.trials) + 1,
            "Timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Query": st.session_state.last_query,
            "Top-K": len(results),
            "Threshold": st.session_state.last_threshold,
            "Documents Compared": st.session_state.last_documents_compared,
            "Best Document": best["Title"],
            "Best Similarity": round(best["Similarity"], 4),
            "Query Time (sec)": round(st.session_state.last_query_time, 4),
        }
    )


# ============================================================
# PDF REPORT
# ============================================================

class LabReportPDF(FPDF):

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Dense Embedding-Based Semantic Search", ln=True, align="C")
        self.set_font("Arial", "", 9)
        self.cell(0, 7, "Virtual Laboratory Experiment Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def safe_pdf_text(value):
    if value is None:
        return ""

    value = str(value)

    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
        "\u00b7": "*",
        "\u2192": "->",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value.encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(student_name, roll_number, department, experiment_date, observations):
    pdf = LabReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def heading(text):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, safe_pdf_text(text), ln=True)
        pdf.set_font("Arial", "", 10)

    heading("Student Information")

    for detail in [
        f"Student Name: {student_name}",
        f"Roll Number: {roll_number}",
        f"Department: {department}",
        f"Experiment Date: {experiment_date}",
    ]:
        pdf.cell(0, 7, safe_pdf_text(detail), ln=True)

    pdf.ln(5)

    heading("Aim")
    pdf.multi_cell(0, 6, safe_pdf_text(THEORY_AIM))
    pdf.ln(4)

    heading("Theory Summary")
    pdf.multi_cell(0, 6, safe_pdf_text(THEORY_INTRODUCTION))
    pdf.ln(4)

    heading("Experimental Setup")

    documents_df = get_active_documents()

    for line in [
        f"Embedding Model: {MODEL_NAME}",
        f"Documents in Collection: {len(documents_df)}",
        f"Categories: {documents_df['category'].nunique()}",
        f"Embedding Dimension: {st.session_state.last_embedding_dimension or 384}",
        f"Index Status: {'Ready' if st.session_state.index_built else 'Not built'}",
        f"Pretest Score: {st.session_state.pretest_score}/{len(st.session_state.pretest_selected or [])}",
        f"Posttest Score: {st.session_state.posttest_score}/{len(st.session_state.posttest_selected or [])}",
    ]:
        pdf.cell(0, 7, safe_pdf_text(line), ln=True)

    pdf.ln(4)

    heading("Observations")
    pdf.multi_cell(0, 6, safe_pdf_text(observations))
    pdf.ln(4)

    heading("Experiment Trials")

    if st.session_state.trials:
        headers = ["Trial", "Query", "Top-K", "Thr.", "Best Document", "Score", "Time"]
        widths = [12, 42, 14, 14, 48, 20, 19]

        pdf.set_font("Arial", "B", 8)
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, safe_pdf_text(header), border=1)
        pdf.ln()

        pdf.set_font("Arial", "", 7)
        for trial in st.session_state.trials:
            values = [
                trial.get("Trial", ""),
                trial.get("Query", ""),
                trial.get("Top-K", ""),
                trial.get("Threshold", ""),
                trial.get("Best Document", ""),
                trial.get("Best Similarity", ""),
                trial.get("Query Time (sec)", ""),
            ]

            for value, width in zip(values, widths):
                text = safe_pdf_text(value)
                if len(text) > 30:
                    text = text[:27] + "..."
                pdf.cell(width, 7, text, border=1)

            pdf.ln()
    else:
        pdf.cell(0, 7, "No trials recorded.", ln=True)

    pdf.ln(5)

    heading("Conclusion")

    conclusion = """
The experiment implemented semantic search using dense sentence
embeddings generated by SentenceTransformer. Documents were indexed
once, and each query was compared against the stored embeddings using
cosine similarity. Ranking, a Top-K limit and a minimum similarity
threshold were used to return the most semantically relevant documents.
"""

    pdf.multi_cell(0, 6, safe_pdf_text(conclusion))

    return bytes(pdf.output())


# ============================================================
# HEADER / BREADCRUMB / NAVIGATION
# ============================================================

def render_top_header():
    render_html(
        """
        <div class="top-header">
            <div class="menu-symbol">&#9776;</div>
            <div>
                <div class="logo-main">Virtual <span class="logo-sub">Labs</span></div>
                <div class="logo-caption">An MoE Government of India Initiative</div>
            </div>
            <div class="header-spacer"></div>
            <div class="rating">&#9733;&#9733;&#9733;&#9733;&#9734;</div>
            <div class="header-button">Rate Me</div>
            <div class="header-button">Report a Bug</div>
        </div>
        <div class="orange-line"></div>
        """
    )


def render_breadcrumb():
    render_html(
        """
        <div class="breadcrumb">
            Computer Science and Engineering
            &nbsp;&rsaquo;&nbsp;
            Natural Language Processing Virtual Laboratory
            &nbsp;&rsaquo;&nbsp;
            Experiments
        </div>
        """
    )


def pipeline_markup(stages, active=None):
    """Pipeline where stages before 'active' are done and the rest are pending."""
    parts = []

    for position, stage in enumerate(stages):
        if active is None:
            state = ""
        elif position < active:
            state = " stage-done"
        elif position == active:
            state = " stage-active"
        else:
            state = " stage-todo"

        if position:
            parts.append('<span class="pipeline-arrow">&rarr;</span>')

        parts.append(
            f'<span class="pipeline-stage{state}">{escape_html(stage)}</span>'
        )

    return '<div class="pipeline">' + "".join(parts) + "</div>"


def render_pipeline(stages, active=None):
    render_html(pipeline_markup(stages, active))


def log_markup(lines):
    body = "<br>".join(lines)
    return f'<div class="live-log">{body}</div>'


INDEX_STAGES = [
    "Documents",
    "Preprocessing",
    "SentenceTransformer",
    "Dense Embeddings",
    "Embedding Index",
]

QUERY_STAGES = [
    "User Query",
    "Preprocessing",
    "Query Embedding",
    "Cosine Similarity",
    "Ranking",
    "Top-K Results",
]


def render_left_navigation():
    render_html('<div class="left-navigation-title">Experiment Sections</div>')

    sections = list(SECTION_RENDERERS.keys())

    active = st.session_state.active_section

    if active in sections:
        active_index = sections.index(active)

        render_html(
            f"""
            <style>
            .st-key-nav_{active_index} button {{
                color: #f47721 !important;
                font-weight: 700 !important;
            }}
            </style>
            """
        )

    for index, section in enumerate(sections):
        with st.container(key=f"nav_{index}"):
            if st.button(section, key=f"navigation_{section}", use_container_width=True):
                st.session_state.active_section = section


# ============================================================
# SECTION: AIM AND INTRODUCTION (merged)
# ============================================================

def render_aim_and_introduction():
    render_html('<div class="content-heading">Aim and Introduction</div>')

    render_html('<div class="content-subheading">Aim of the experiment</div>')

    render_html(
        """
        <div class="aim-list">
            <ul>
                <li>To implement semantic search using dense embeddings.</li>
                <li>To convert a document collection and a query into numerical vectors.</li>
                <li>To build, inspect and reuse an embedding index.</li>
                <li>To calculate cosine similarity between embeddings.</li>
                <li>To retrieve the most semantically relevant documents.</li>
                <li>To study the effect of the Top-K parameter and the similarity threshold.</li>
            </ul>
        </div>
        """
    )

    render_html('<div class="content-subheading">Objectives</div>')

    for objective in THEORY_OBJECTIVES:
        st.markdown(f"- {objective}")

    render_html('<div class="content-subheading">Introduction</div>')

    render_html(
        f'<div class="content-text">{to_html_paragraphs(THEORY_INTRODUCTION)}</div>'
    )

    render_html('<div class="content-subheading">Why Semantic Search?</div>')

    st.write(
        """
        Keyword search depends mainly on matching exact words. Semantic search
        understands the meaning of the query and can retrieve related documents
        even when the wording is completely different. This experiment builds
        such a system end to end and shows every intermediate step, from raw
        text to dense vectors to a ranked result list.
        """
    )

    comparison = pd.DataFrame(
        [
            {
                "Aspect": "Matching basis",
                "Keyword Search": "Exact terms shared by query and document",
                "Semantic Search": "Closeness of meaning in a vector space",
            },
            {
                "Aspect": "Representation",
                "Keyword Search": "Sparse high-dimensional term vectors",
                "Semantic Search": "Dense fixed-length embeddings",
            },
            {
                "Aspect": "Vocabulary mismatch",
                "Keyword Search": "Fails when wording differs",
                "Semantic Search": "Handled, paraphrases still match",
            },
            {
                "Aspect": "Rare exact tokens",
                "Keyword Search": "Handled very well",
                "Semantic Search": "Can be missed",
            },
            {
                "Aspect": "Cost",
                "Keyword Search": "Cheap index, cheap query",
                "Semantic Search": "Model inference needed for indexing and querying",
            },
        ]
    )

    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ============================================================
# SECTION: THEORY AND APPLICATION (merged)
# ============================================================

def render_theory_and_application():
    """Render the expanded theory section without changing the Simulation section."""
    render_html('<div class="content-heading">Theory and Application</div>')

    # ------------------------------------------------------------------
    # 1. Keyword search versus dense semantic search
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">Keyword Search vs. Semantic Search</div>')

    left_col, right_col = st.columns(2)

    with left_col:
        render_html(
            """
            <div class="result-card" style="border-left-color:#e05252; background:#fff7f7;">
                <div class="result-title" style="color:#b83232; font-size:20px;">
                    Traditional Keyword Matching
                </div>
                <div class="result-content">
                    <b>Matching basis:</b> Exact keyword overlap between the query and the document.
                    <br><br>
                    <b>Example query:</b> “comfortable shoes for running”
                    <br><br>
                    The system searches for words such as <b>comfortable</b>, <b>shoes</b>,
                    <b>for</b> and <b>running</b>.
                    <br><br>
                    <span style="color:#b83232;"><b>Limitations:</b></span>
                    <ul>
                        <li>May miss relevant products or documents with different wording.</li>
                        <li>Depends strongly on exact terms present in the index.</li>
                        <li>Has limited understanding of user intent and meaning.</li>
                    </ul>
                </div>
            </div>
            """
        )

    with right_col:
        render_html(
            """
            <div class="result-card" style="border-left-color:#2f9e68; background:#f4fff8;">
                <div class="result-title" style="color:#187344; font-size:20px;">
                    Dense Semantic Search
                </div>
                <div class="result-content">
                    <b>Matching basis:</b> Semantic similarity between dense vector embeddings.
                    <br><br>
                    <b>Example query:</b> “comfortable shoes for running”
                    <br><br>
                    The model converts the query and documents into numerical vectors and
                    compares their meaning using cosine similarity.
                    <br><br>
                    <span style="color:#187344;"><b>Advantages:</b></span>
                    <ul>
                        <li>Finds conceptually similar documents even when wording differs.</li>
                        <li>Captures contextual meaning and user intent.</li>
                        <li>Ranks results according to semantic relevance.</li>
                    </ul>
                </div>
            </div>
            """
        )

    # ------------------------------------------------------------------
    # 2. Embedding fundamentals and process
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">What Is an Embedding?</div>')
    render_html(
        """
        <div class="content-text">
            An <b>embedding</b> converts human text into a fixed-length numerical vector
            so that mathematical algorithms can compare the meaning of different pieces
            of text. Sentences with similar meanings generally produce vectors that point
            in similar directions in the learned vector space.
        </div>
        """
    )

    render_pipeline(
        [
            "Raw Text",
            "Tokenization",
            "Transformer Encoder",
            "Mean Pooling",
            "384-D Dense Vector",
        ]
    )

    render_html(
        """
        <div class="info-box">
            <b>Key idea:</b> The same pretrained sentence-transformer model is used for
            both documents and queries. This places them in a common 384-dimensional
            vector space where their similarity can be computed directly.
        </div>
        """
    )

    # ------------------------------------------------------------------
    # 3. Model theory
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">The all-MiniLM-L6-v2 Model</div>')
    st.markdown(
        """
        This experiment uses **all-MiniLM-L6-v2** from the Sentence Transformers
        library. It is a compact sentence-transformer model that produces a
        384-dimensional representation for each input text.
        """
    )

    model_table = pd.DataFrame(
        [
            ["Architecture", "MiniLM (distilled BERT-style Transformer)", "Six Transformer layers with attention"],
            ["Output dimension", "384", "Each text becomes 384 floating-point values"],
            ["Training objective", "Sentence-pair similarity learning", "Related sentences are represented closer together"],
            ["Similarity metric", "Cosine similarity", "Measures the angle between two vectors"],
            ["Library", "sentence-transformers", "Provides pretrained sentence-embedding models"],
        ],
        columns=["Property", "Value", "Description"],
    )
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    render_html(
        """
        <div class="info-box">
            <b>Why this model?</b> all-MiniLM-L6-v2 provides compact sentence embeddings
            that are suitable for interactive experiments because the vectors are small
            enough to compute quickly while retaining useful semantic information.
        </div>
        """
    )

    # ------------------------------------------------------------------
    # 4. Dense embeddings and document/query vectors
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">Dense Embeddings</div>')
    st.markdown(
        """
        A **dense embedding** contains mostly non-zero numerical values, unlike sparse
        representations such as Bag-of-Words or TF-IDF. Dense embeddings are generated
        by neural networks trained on large text corpora and can capture contextual and
        conceptual relationships.

        The individual dimensions do not have isolated human-interpretable labels.
        Instead, semantic meaning emerges from the collective pattern across all
        384 dimensions.
        """
    )

    doc_col, query_col = st.columns(2)
    with doc_col:
        render_html(
            """
            <div class="result-card">
                <div class="result-title" style="font-size:19px;">Document Embeddings</div>
                <div class="result-content">
                    Each document is encoded into a 384-dimensional vector before retrieval
                    begins. These vectors form the searchable embedding index and are reused
                    for future queries.
                </div>
            </div>
            """
        )
    with query_col:
        render_html(
            """
            <div class="result-card">
                <div class="result-title" style="font-size:19px;">Query Embedding</div>
                <div class="result-content">
                    Whenever the user enters a query, the same model converts it into a
                    384-dimensional vector. This query vector is compared with the stored
                    document vectors using cosine similarity.
                </div>
            </div>
            """
        )

    # ------------------------------------------------------------------
    # 5. Figures supplied for the theory section
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">Visual Explanation of the Concept</div>')

    figures = [
        ("figure1_keyword_vs_semantic.jpeg",
         "Figure 1: Traditional Keyword Matching vs. Dense Semantic Search in E-Commerce Intent Retrieval"),
        ("fig2.jpeg",
         "Figure 2: Text-to-Dense Embedding Process using Pretrained Transformer Models"),
        ("fig3.jpeg",
         "Figure 3: Geometric Interpretation of Cosine Similarity in High-Dimensional Vector Space"),
        ("fig4.jpeg",
         "Figure 4: Disadvantages and Limitations of Dense Semantic Search"),
    ]

    for filename, caption in figures:
        figure_path = THEORY_FIGURE_DIR / filename
        if figure_path.exists():
            st.image(str(figure_path), use_container_width=True)
            st.caption(caption)
        else:
            st.warning(f"Theory figure is missing: {figure_path}")

    # ------------------------------------------------------------------
    # 6. Cosine similarity
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">Cosine Similarity</div>')
    st.markdown(
        """
        Cosine similarity calculates the cosine of the angle θ between a query vector
        **A** and a document vector **B**. It measures the orientation of the vectors,
        rather than their absolute magnitude.
        """
    )
    st.latex(r"\text{Cosine Similarity}(A,B) = \frac{A \cdot B}{\|A\|\|B\|} = \cos(\theta)")

    st.markdown(
        """
        In this experiment, embeddings are normalised to unit length. Therefore, the
        denominator becomes one and cosine similarity reduces to the dot product:

        **Similarity(A, B) = A · B**
        """
    )

    cosine_table = pd.DataFrame(
        [
            ["+1", "Vectors point in the same direction", "Very high semantic similarity"],
            ["Around 0", "Vectors are approximately orthogonal", "Little or no semantic correlation"],
            ["-1", "Vectors point in opposite directions", "Opposite orientation in vector space"],
        ],
        columns=["Cosine value", "Geometric interpretation", "General meaning"],
    )
    st.dataframe(cosine_table, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 7. Retrieval pipeline and ranking
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">The Two-Stage Retrieval Pipeline</div>')
    st.markdown("**Stage A — Document Indexing (performed once)**")
    render_pipeline(INDEX_STAGES)

    st.markdown("**Stage B — Query Search (performed for each query)**")
    render_pipeline(QUERY_STAGES)

    render_html('<div class="content-subheading">Similarity Ranking and Top-K Retrieval</div>')
    st.markdown(
        """
        After cosine similarity is calculated for the query against all candidate
        documents:

        1. Documents are sorted in descending order of similarity score.
        2. **Top-K retrieval** selects the K highest-ranked documents.
        3. The similarity threshold can remove documents whose score is too low.

        **Important distinction:** Top-K represents the number of returned documents,
        whereas **384** represents the embedding dimension. These are independent
        parameters.
        """
    )

    # ------------------------------------------------------------------
    # 8. Important terms
    # ------------------------------------------------------------------
    render_html('<div class="content-subheading">Important Terms</div>')
    terms = pd.DataFrame(
        [
            ["Embedding", "Numerical vector representation of text."],
            ["Dense vector", "Fixed-length vector whose values are mostly non-zero."],
            ["Transformer", "Neural architecture based on self-attention."],
            ["Mean pooling", "Averaging token vectors into one sentence vector."],
            ["Embedding index", "Stored matrix of document embeddings reused for every query."],
            ["Cosine similarity", "Similarity based on the angle between two vectors."],
            ["Top-K", "Number of highest-ranked documents returned."],
            ["Similarity threshold", "Minimum score a document must reach to be shown."],
            ["Vector database", "System that stores embeddings and supports similarity search."],
            ["Semantic search", "Retrieval based on meaning instead of exact keywords."],
        ],
        columns=["Term", "Meaning"],
    )
    st.dataframe(terms, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 9. Application and limitations
    # ------------------------------------------------------------------
    render_html('<div class="content-heading">Application</div>')
    st.markdown(
        """
        Dense embedding-based retrieval is used in modern search and assistant
        systems. It is useful when the user and the stored content may express
        the same idea using different words.
        """
    )

    for position in range(0, len(APPLICATION_AREAS), 2):
        columns = st.columns(2)
        for column, (title, description) in zip(
            columns, APPLICATION_AREAS[position:position + 2]
        ):
            with column:
                render_html(
                    f"""
                    <div class="result-card">
                        <div class="result-title" style="font-size:19px;">
                            {escape_html(title)}
                        </div>
                        <div class="result-content">
                            {escape_html(description)}
                        </div>
                    </div>
                    """
                )

    render_html('<div class="content-subheading">Disadvantages and Limitations</div>')
    limitations = [
        ("Computational cost", "Generating dense embeddings for large collections requires more computation and may benefit from GPU acceleration."),
        ("Model dependence", "Retrieval quality depends on the training data, domain suitability and capabilities of the selected embedding model."),
        ("Semantic ambiguity", "Similar-looking embeddings do not always guarantee that two documents are truly relevant. Polysemous or negated phrases can cause problems."),
        ("Domain limitations", "A general-purpose model may perform poorly on highly specialised technical, medical, legal or domain-specific content without adaptation."),
        ("No exact-match guarantee", "Semantic search may miss exact keywords, part numbers, codes, dates or specific acronyms that lexical search can retrieve effectively."),
    ]

    for title, description in limitations:
        render_html(
            f"""
            <div class="info-box">
                <b>{escape_html(title)}:</b> {escape_html(description)}
            </div>
            """
        )

    render_html(
        """
        <div class="info-box">
            <b>Practical note:</b> Production retrieval systems often combine dense
            retrieval with a lexical method such as BM25 and may use a cross-encoder
            re-ranker. Dense retrieval supports semantic recall, while lexical
            matching helps preserve exact terms and identifiers.
        </div>
        """
    )

    render_html('<div class="content-subheading">Why Does This Work?</div>')
    compare_left, compare_right = st.columns(2)

    with compare_left:
        render_html(
            """
            <div class="result-card" style="border-left-color:#f47721;">
                <div class="result-title" style="font-size:19px;">Traditional Search</div>
                <div class="result-content">
                    Text → keyword matching → results
                    <br><br>
                    The system may fail when synonyms or paraphrases are used.
                </div>
            </div>
            """
        )

    with compare_right:
        render_html(
            """
            <div class="result-card" style="border-left-color:#198754;">
                <div class="result-title" style="font-size:19px;">Semantic Search</div>
                <div class="result-content">
                    Text → embeddings → similarity → ranking → results
                    <br><br>
                    Related concepts can be represented by vectors that are close
                    in the learned embedding space.
                </div>
            </div>
            """
        )

    render_html(
        """
        <div class="info-box">
            <b>Key takeaway:</b> Words are converted into vectors by a pretrained
            sentence-transformer model. Text with related meaning can be represented
            by vectors that are closer in the learned embedding space. Retrieval
            then becomes a vector similarity problem.
        </div>
        """
    )


# ============================================================
# QUIZ RENDERING
# ============================================================
#
# Each test has a bank of 50 questions. Every user session draws its own
# random sample of QUESTIONS_PER_ATTEMPT questions from that bank -- picked
# once per session and then held fixed in session state, so the same 10
# questions stay in front of the user while they answer and after they
# submit, but the next person (a new session) gets a different random 10.

QUESTIONS_PER_ATTEMPT = 10


def get_quiz_selection(bank, prefix, count=QUESTIONS_PER_ATTEMPT):
    key = f"{prefix}_selected"

    if not st.session_state.get(key):
        st.session_state[key] = random.sample(range(len(bank)), min(count, len(bank)))

    return st.session_state[key]


def reshuffle_quiz(bank, prefix, count=QUESTIONS_PER_ATTEMPT):
    for index in st.session_state.get(f"{prefix}_selected") or []:
        st.session_state.pop(f"{prefix}_answer_{index}", None)

    st.session_state[f"{prefix}_selected"] = random.sample(
        range(len(bank)), min(count, len(bank))
    )
    st.session_state[f"{prefix}_submitted"] = False
    st.session_state[f"{prefix}_score"] = 0
    st.session_state[f"{prefix}_attempted"] = 0


def render_quiz(bank, prefix, heading, intro):
    render_html(f'<div class="content-heading">{heading}</div>')
    st.write(intro)

    selected = get_quiz_selection(bank, prefix)
    total = len(selected)

    answered = sum(
        1 for index in selected if st.session_state.get(f"{prefix}_answer_{index}")
    )

    render_html(
        f'<div class="quiz-progress">Questions attempted: '
        f"{answered} of {total} (drawn randomly from a bank of {len(bank)})</div>"
    )

    st.progress(answered / total)

    for position, index in enumerate(selected, start=1):
        question = bank[index]

        options = question["options"]
        shift = index % len(options)
        ordered = options[shift:] + options[:shift]

        st.markdown(f"**Q{position}. {question['question']}**")

        st.radio(
            "Select an option:",
            ordered,
            index=None,
            key=f"{prefix}_answer_{index}",
            label_visibility="collapsed",
        )

        if st.session_state.get(f"{prefix}_submitted"):
            chosen = st.session_state.get(f"{prefix}_answer_{index}")

            if chosen is None:
                st.warning(f"Not attempted. Correct answer: {question['answer']}")
            elif chosen == question["answer"]:
                st.success("Correct.")
            else:
                st.error(f"Incorrect. Correct answer: {question['answer']}")

        st.markdown("---")

    action_col, reset_col, reshuffle_col = st.columns(3)

    with action_col:
        if st.button("Submit and Evaluate", key=f"{prefix}_submit", use_container_width=True):
            score = 0
            attempted = 0

            for index in selected:
                chosen = st.session_state.get(f"{prefix}_answer_{index}")

                if chosen is not None:
                    attempted += 1

                    if chosen == bank[index]["answer"]:
                        score += 1

            st.session_state[f"{prefix}_score"] = score
            st.session_state[f"{prefix}_attempted"] = attempted
            st.session_state[f"{prefix}_submitted"] = True

    with reset_col:
        if st.button("Clear My Answers", key=f"{prefix}_reset", use_container_width=True):
            for index in selected:
                st.session_state.pop(f"{prefix}_answer_{index}", None)

            st.session_state[f"{prefix}_submitted"] = False
            st.session_state[f"{prefix}_score"] = 0
            st.session_state[f"{prefix}_attempted"] = 0

    with reshuffle_col:
        if st.button("New Random Set of 10", key=f"{prefix}_reshuffle", use_container_width=True):
            reshuffle_quiz(bank, prefix)
            st.rerun()

    if st.session_state.get(f"{prefix}_submitted"):
        score = st.session_state[f"{prefix}_score"]
        attempted = st.session_state[f"{prefix}_attempted"]

        render_html(
            f"""
            <div class="result-box">
                <b>Result</b><br>
                Score: {score} out of {total}<br>
                Attempted: {attempted} of {total}<br>
                Percentage: {score / total * 100:.2f}%
            </div>
            """
        )


def render_pretest():
    render_quiz(
        PRETEST_QUESTIONS,
        "pretest",
        "Pretest",
        f"This pretest draws {QUESTIONS_PER_ATTEMPT} questions at random from a "
        f"bank of {len(PRETEST_QUESTIONS)}, so different attempts (and different "
        "users) usually see a different set. Your selection stays fixed for this "
        "session unless you press \"New Random Set of 10\".",
    )


def render_posttest():
    render_quiz(
        POSTTEST_QUESTIONS,
        "posttest",
        "Posttest",
        f"This posttest draws {QUESTIONS_PER_ATTEMPT} questions at random from a "
        f"bank of {len(POSTTEST_QUESTIONS)} covering indexing, similarity, ranking, "
        "evaluation and the behaviour you observed in the simulation. Your selection "
        "stays fixed for this session unless you press \"New Random Set of 10\".",
    )


def render_procedure():
    render_html('<div class="content-heading">Procedure</div>')

    for index, step in enumerate(PROCEDURE_STEPS, start=1):
        render_html(
            f'<div class="content-text"><b>{index}.</b> {escape_html(step)}</div>'
        )

    render_html('<div class="content-subheading">Input</div>')
    st.write("A document collection and a natural language query entered by the user.")

    render_html('<div class="content-subheading">Output</div>')
    st.write(
        "Ranked documents with cosine similarity scores, filtered by the "
        "Top-K limit and the minimum similarity threshold."
    )


# ============================================================
# SIMULATION: DATASET PANEL
# ============================================================

def render_dataset_panel(documents_df, source_label):
    render_html('<div class="content-subheading">1. Document Collection</div>')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", len(documents_df))
    col2.metric("Categories", documents_df["category"].nunique())
    col3.metric("Average Words", int(documents_df["word_count"].mean()))
    col4.metric("Sources", documents_df["source"].nunique())

    st.caption(f"Dataset loaded from: {source_label}")

    with st.expander("Browse the document collection", expanded=False):
        categories = ["All categories"] + sorted(documents_df["category"].unique().tolist())

        selected = st.selectbox("Filter by category", categories, key="dataset_category_filter")
        keyword = st.text_input("Filter by keyword in title or content", key="dataset_keyword_filter")

        view = documents_df

        if selected != "All categories":
            view = view[view["category"] == selected]

        if keyword.strip():
            mask = view["title"].str.contains(keyword, case=False, na=False) | view[
                "content"
            ].str.contains(keyword, case=False, na=False)
            view = view[mask]

        st.caption(f"Showing {len(view)} of {len(documents_df)} documents.")
        st.dataframe(view, use_container_width=True, hide_index=True, height=340)

        st.download_button(
            "Download Document Collection (CSV)",
            data=documents_df.to_csv(index=False),
            file_name="document_collection.csv",
            mime="text/csv",
        )

    with st.expander("Add your own documents", expanded=False):
        st.write(
            "Upload a CSV (columns: id, title, category, content), a JSON array of "
            "objects, or a TXT file where each blank-line-separated block is one "
            "document. Uploaded documents are appended to the default collection."
        )

        uploaded_file = st.file_uploader(
            "Upload a document dataset",
            type=["csv", "txt", "json"],
            key="dataset_uploader",
        )

        upload_col, clear_col = st.columns(2)

        with upload_col:
            if st.button("Add Uploaded Documents", use_container_width=True):
                if uploaded_file is None:
                    st.warning("Please choose a file first.")
                else:
                    try:
                        parsed = parse_uploaded_file(uploaded_file)
                        st.session_state.uploaded_documents = parsed
                        st.session_state.index_built = False
                        st.session_state.upload_message = (
                            f"Added {len(parsed)} documents from {uploaded_file.name}. "
                            "Rebuild the embedding index to search them."
                        )
                    except Exception as error:
                        st.session_state.upload_message = ""
                        st.error(f"Could not read the file: {error}")

        with clear_col:
            if st.button("Remove Uploaded Documents", use_container_width=True):
                st.session_state.uploaded_documents = None
                st.session_state.index_built = False
                st.session_state.upload_message = "Uploaded documents removed."

        if st.session_state.upload_message:
            st.info(st.session_state.upload_message)


# ============================================================
# SIMULATION: STAGE A WITH LIVE VISUALISATION
# ============================================================

def run_live_indexing(documents_df, show_progress, pace=0.25, batch_size=16):
    """Encode the collection batch by batch. A clean animated status replaces
    the previous raw text log -- just the pipeline lighting up, a progress bar
    and a one-line caption, which is what actually reads as "happening" to a
    viewer rather than as backend output. `pace` sets the pause (in seconds)
    between narrated steps, so the caller can speed this up or slow it down."""
    pipeline_placeholder = st.empty()
    caption_placeholder = st.empty()
    progress_placeholder = st.empty()

    def stage(active):
        if show_progress:
            pipeline_placeholder.markdown(
                pipeline_markup(INDEX_STAGES, active), unsafe_allow_html=True
            )

    start = time.perf_counter()

    stage(0)
    if show_progress:
        caption_placeholder.caption(f"Reading {len(documents_df)} documents ...")
        time.sleep(pace)

    stage(1)
    texts = documents_df["content"].astype(str).tolist()
    if show_progress:
        caption_placeholder.caption("Preprocessing text (tokenizing, cleaning) ...")
        time.sleep(pace)

    stage(2)
    if show_progress:
        caption_placeholder.caption(f"Loading the {MODEL_NAME} model ...")

    model = load_embedding_model()

    stage(3)
    progress = progress_placeholder.progress(0.0) if show_progress else None

    chunks = []
    encoded = 0

    for start_index in range(0, len(texts), batch_size):
        batch = texts[start_index:start_index + batch_size]

        vectors = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        chunks.append(vectors)
        encoded += len(batch)

        if show_progress:
            progress.progress(encoded / len(texts))
            caption_placeholder.caption(
                f"Encoding documents into vectors ... {encoded}/{len(texts)}"
            )

    embeddings = np.vstack(chunks)

    stage(4)
    elapsed = time.perf_counter() - start

    if show_progress:
        caption_placeholder.caption(
            f"Embedding index built: {embeddings.shape[0]} vectors x "
            f"{embeddings.shape[1]} dimensions in {elapsed:.2f} s."
        )
        progress_placeholder.empty()

    store_index(embeddings, documents_df, elapsed)

    return embeddings


def render_embedding_map_animated(embeddings, documents_df):
    """An actually-animated Plotly scatter: categories reveal one group at a
    time with a Play button, so the viewer watches clusters form instead of
    seeing a finished plot appear all at once."""
    points = pca_projection(embeddings)

    frame_df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "Title": documents_df["title"],
            "Category": documents_df["category"],
        }
    )

    categories = list(dict.fromkeys(frame_df["Category"].tolist()))
    palette = px.colors.qualitative.Set2
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    x_pad = (frame_df["x"].max() - frame_df["x"].min()) * 0.1 + 0.01
    y_pad = (frame_df["y"].max() - frame_df["y"].min()) * 0.1 + 0.01
    x_range = [frame_df["x"].min() - x_pad, frame_df["x"].max() + x_pad]
    y_range = [frame_df["y"].min() - y_pad, frame_df["y"].max() + y_pad]

    frames = []
    shown_categories = []

    for category in categories:
        shown_categories.append(category)
        visible = frame_df[frame_df["Category"].isin(shown_categories)]

        frame_traces = [
            go.Scatter(
                x=visible.loc[visible["Category"] == cat, "x"],
                y=visible.loc[visible["Category"] == cat, "y"],
                mode="markers",
                name=cat,
                marker={"size": 10, "color": color_map[cat], "opacity": 0.85},
                text=visible.loc[visible["Category"] == cat, "Title"],
                hoverinfo="text",
            )
            for cat in categories
        ]

        frames.append(go.Frame(data=frame_traces, name=category))

    initial_traces = [
        go.Scatter(x=[], y=[], mode="markers", name=cat, marker={"color": color_map[cat]})
        for cat in categories
    ]

    figure = go.Figure(
        data=initial_traces,
        frames=frames,
        layout=go.Layout(
            title="Embedding space (documents appear category by category)",
            xaxis={"title": "Component 1", "range": x_range},
            yaxis={"title": "Component 2", "range": y_range},
            height=470,
            updatemenus=[
                {
                    "type": "buttons",
                    "x": 0,
                    "y": 1.12,
                    "buttons": [
                        {
                            "label": "&#9654; Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 650, "redraw": True},
                                    "transition": {"duration": 200},
                                    "fromcurrent": False,
                                },
                            ],
                        }
                    ],
                }
            ],
        ),
    )

    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Press Play to watch each category's documents land in the space. "
        "Documents about similar topics cluster together -- that closeness is "
        "exactly what cosine similarity measures at query time."
    )


def render_indexing_panel(documents_df, show_progress, pace=0.25):
    render_html('<div class="stage-label">STAGE A</div>')
    render_html('<div class="content-subheading">2. Build the Embedding Index</div>')

    if not index_is_current(documents_df):
        render_html(
            """
            <div class="info-box">
                The embedding index is not up to date with the current document
                collection. Build the index before running a search.
            </div>
            """
        )
        render_pipeline(INDEX_STAGES, 0)

    if st.button("Build Embedding Index", use_container_width=True):
        run_live_indexing(documents_df, show_progress, pace=pace)
        st.session_state.last_results = []
        st.success("Embedding index created successfully.")

    if index_is_current(documents_df):
        embeddings = st.session_state.document_embeddings

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents Indexed", embeddings.shape[0])
        col2.metric("Embedding Dimension", embeddings.shape[1])
        col3.metric("Index Build Time", f"{st.session_state.index_build_time:.2f} sec")
        col4.metric("Index Status", "Ready")

        with st.expander("View Embedding Details"):
            st.write(f"**Model:** {MODEL_NAME}")
            st.write(f"**Number of documents:** {embeddings.shape[0]}")
            st.write(f"**Embedding dimension:** {embeddings.shape[1]}")
            st.write("**Embedding type:** Dense floating-point vector (L2 normalised)")
            st.write(f"**Index memory:** {embeddings.nbytes / 1024:.1f} KB")

            st.caption(
                f"First 5 documents, first 16 of {embeddings.shape[1]} dimensions."
            )

            st.dataframe(
                pd.DataFrame(
                    embeddings[:5, :16].round(4),
                    index=st.session_state.indexed_documents["id"].head(5),
                    columns=[f"d{i}" for i in range(16)],
                ),
                use_container_width=True,
            )

        with st.expander("Watch the embedding space form (animated)", expanded=False):
            render_embedding_map_animated(embeddings, st.session_state.indexed_documents)


# ============================================================
# SIMULATION: STAGE B WITH LIVE VISUALISATION
# ============================================================

def run_live_search(query, documents_df, embeddings, top_k, threshold, show_progress, pace=0.3):
    """Stage B, with the pipeline strip lighting up stage by stage instead of
    a scrolling text log. `pace` sets the pause between narrated steps."""
    pipeline_placeholder = st.empty()
    caption_placeholder = st.empty()

    def stage(active, caption=""):
        if show_progress:
            pipeline_placeholder.markdown(
                pipeline_markup(QUERY_STAGES, active), unsafe_allow_html=True
            )
            if caption:
                caption_placeholder.caption(caption)
            time.sleep(pace)

    start = time.perf_counter()

    stage(0, f'Query received: "{query}"')
    stage(1, "Preprocessing the query text ...")
    stage(2, "Encoding the query with the same model used for the documents ...")

    query_vector, scores = compute_similarities(query, documents_df, embeddings)

    stage(3, f"Computing cosine similarity against {len(documents_df)} documents ...")

    above = int((scores >= threshold).sum())
    stage(4, f"Ranking by score. {above} of {len(scores)} documents pass the threshold.")

    results = assemble_results(scores, documents_df, top_k, threshold)
    elapsed = time.perf_counter() - start

    stage(len(QUERY_STAGES) - 1, f"Returning the top {len(results)} documents.")

    if show_progress:
        pipeline_placeholder.markdown(
            pipeline_markup(QUERY_STAGES, len(QUERY_STAGES)), unsafe_allow_html=True
        )

    return results, elapsed, int(query_vector.shape[0]), scores


def render_similarity_radar(results):
    """An animated Plotly chart: the query sits at the centre and each result
    starts on the outer ring, then pulls in toward the centre by exactly how
    similar it is. Press Play to watch closeness happen -- an intuitive,
    animated stand-in for what cosine similarity means."""
    if not results:
        return

    count = len(results)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)

    start_x = np.cos(angles)
    start_y = np.sin(angles)

    end_radius = np.array([max(0.05, 1 - r["Similarity"]) for r in results])
    end_x = end_radius * np.cos(angles)
    end_y = end_radius * np.sin(angles)

    titles = [r["Title"] for r in results]
    sims = [r["Similarity"] for r in results]
    hover = [f"{t}<br>similarity {s:.3f}" for t, s in zip(titles, sims)]

    ring_theta = np.linspace(0, 2 * np.pi, 100)

    start_frame = go.Frame(
        data=[
            go.Scatter(
                x=np.cos(ring_theta), y=np.sin(ring_theta),
                mode="lines", line={"color": "#dddddd", "dash": "dot"}, showlegend=False,
            ),
            go.Scatter(x=[0], y=[0], mode="markers+text", text=["Your query"],
                       textposition="bottom center", marker={"size": 18, "color": "#f47721"},
                       showlegend=False),
            go.Scatter(x=start_x, y=start_y, mode="markers+text",
                       text=[f"#{i + 1}" for i in range(count)], textposition="top center",
                       marker={"size": 13, "color": "#2696d2"},
                       hovertext=hover, hoverinfo="text", showlegend=False),
        ],
        name="start",
    )

    end_frame = go.Frame(
        data=[
            go.Scatter(
                x=np.cos(ring_theta), y=np.sin(ring_theta),
                mode="lines", line={"color": "#dddddd", "dash": "dot"}, showlegend=False,
            ),
            go.Scatter(x=[0], y=[0], mode="markers+text", text=["Your query"],
                       textposition="bottom center", marker={"size": 18, "color": "#f47721"},
                       showlegend=False),
            go.Scatter(x=end_x, y=end_y, mode="markers+text",
                       text=[f"#{i + 1}" for i in range(count)], textposition="top center",
                       marker={"size": 13, "color": "#198754"},
                       hovertext=hover, hoverinfo="text", showlegend=False),
        ],
        name="end",
    )

    figure = go.Figure(
        data=start_frame.data,
        frames=[start_frame, end_frame],
        layout=go.Layout(
            title="How close each result is to your query (press Play)",
            xaxis={"visible": False, "range": [-1.2, 1.2]},
            yaxis={"visible": False, "range": [-1.2, 1.2], "scaleanchor": "x"},
            height=430,
            showlegend=False,
            updatemenus=[
                {
                    "type": "buttons",
                    "x": 0,
                    "y": 1.12,
                    "buttons": [
                        {
                            "label": "&#9654; Play",
                            "method": "animate",
                            "args": [
                                ["end"],
                                {
                                    "frame": {"duration": 1200, "redraw": True},
                                    "transition": {"duration": 1200, "easing": "cubic-in-out"},
                                    "fromcurrent": False,
                                },
                            ],
                        },
                        {
                            "label": "&#8635; Reset",
                            "method": "animate",
                            "args": [
                                ["start"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
        ),
    )

    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Every result starts on the outer ring. Pressing Play pulls each one "
        "toward your query by exactly its similarity score -- the closer it "
        "ends up, the more relevant it is."
    )


def render_search_panel(documents_df, show_progress, pace=0.3):
    render_html('<div class="stage-label">STAGE B</div>')
    render_html('<div class="content-subheading">3. Run a Query</div>')

    query = st.text_area(
        "Enter your search query",
        value=st.session_state.last_query,
        placeholder="Example: How do machines learn from data?",
        height=100,
    )

    col1, col2 = st.columns(2)

    with col1:
        top_k = st.slider("Number of results to retrieve (Top-K)", 1, 20, 5)

    with col2:
        threshold = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.25, 0.05)

    if st.button("Run Semantic Search", use_container_width=True):
        if not index_is_current(documents_df):
            st.warning("Build the embedding index first (Stage A).")

        elif not query.strip():
            st.warning("Please enter a query.")

        else:
            results, elapsed, dimension, scores = run_live_search(
                query,
                st.session_state.indexed_documents,
                st.session_state.document_embeddings,
                top_k,
                threshold,
                show_progress,
                pace=pace,
            )

            st.session_state.last_query = query
            st.session_state.last_results = results
            st.session_state.last_query_time = elapsed
            st.session_state.last_embedding_dimension = dimension
            st.session_state.last_documents_compared = len(scores)
            st.session_state.last_threshold = threshold
            st.session_state.last_score_distribution = scores

            if not results:
                st.warning(
                    "No document reached the similarity threshold. "
                    "Lower the threshold or rephrase the query."
                )


def render_results_panel():
    results = st.session_state.last_results

    if not results:
        return

    render_html('<div class="content-subheading">4. Query Processing Summary</div>')

    st.info(f"Query received: {st.session_state.last_query}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Query Embedding Dimension", st.session_state.last_embedding_dimension)
    col2.metric("Documents Compared", st.session_state.last_documents_compared)
    col3.metric("Search Time", f"{st.session_state.last_query_time * 1000:.2f} ms")
    col4.metric("Documents Returned", len(results))

    render_html('<div class="content-subheading">5. Ranked Results</div>')

    results_df = pd.DataFrame(results)

    table_df = results_df[
        ["Rank", "Document ID", "Title", "Category", "Source", "Word Count", "Similarity"]
    ].copy()
    table_df["Similarity"] = table_df["Similarity"].round(4)

    with st.expander("View results as a table"):
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    chart_df = results_df.copy()
    chart_df["Document"] = chart_df["Rank"].astype(str) + ". " + chart_df["Title"]
    chart_df = chart_df.sort_values("Similarity").reset_index(drop=True)
    chart_df["Score"] = chart_df["Similarity"].round(4)

    figure = px.bar(
        chart_df,
        x="Similarity",
        y="Document",
        orientation="h",
        title="Cosine Similarity of Retrieved Documents",
        text="Score",
        labels={"Similarity": "Cosine Similarity", "Document": "Document"},
    )

    figure.update_layout(height=430, xaxis_range=[0, 1])

    st.plotly_chart(figure, use_container_width=True)

    with st.expander("Watch the results converge toward your query (animated)", expanded=False):
        render_similarity_radar(results)

    for result in results:
        snippet = result["Content"]
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."

        render_html(
            f"""
            <div class="result-card">
                <div class="result-rank">#{result["Rank"]}</div>
                <div class="result-title">{escape_html(result["Title"])}</div>
                <div class="result-category">{escape_html(result["Category"])}</div>
                <div class="result-content">{escape_html(snippet)}</div>
                <div class="result-score">Cosine Similarity: {result["Similarity"]:.4f}</div>
                <div class="result-meta">
                    Document ID: {escape_html(result["Document ID"])}
                    &nbsp;|&nbsp; Source: {escape_html(result["Source"])}
                    &nbsp;|&nbsp; Words: {result["Word Count"]}
                </div>
            </div>
            """
        )

    if st.button("Record Current Trial", use_container_width=True):
        record_trial()
        st.success("Trial recorded successfully.")


def render_simulation():
    render_html('<div class="content-heading">Simulation</div>')

    render_concept_animation()

    render_html('<hr style="border:none;border-top:1px solid #e5e5e5;margin:28px 0;">')

    render_html('<div class="content-subheading">Try It on the Real Dataset</div>')

    render_html(
        """
        <div class="info-box">
            This part runs the same process on the actual document collection and
            your own queries. Stage A builds the embedding index; Stage B searches it.
        </div>
        """
    )

    col_toggle, col_speed = st.columns([2, 1])

    with col_toggle:
        show_progress = st.checkbox(
            "Show the pipeline animation while indexing and searching",
            value=True,
        )

    with col_speed:
        speed_label = st.select_slider(
            "Animation speed",
            options=["0.5x (Slow)", "1x (Normal)", "1.5x", "2x (Fast)", "3x (Very Fast)"],
            value="1x (Normal)",
            disabled=not show_progress,
        )

    speed_multiplier = float(speed_label.split("x")[0])
    pace = 0.3 / speed_multiplier

    documents_df = get_active_documents()
    _, source_label = load_base_documents()

    if st.session_state.uploaded_documents is not None:
        source_label = f"{source_label} + uploaded file"

    render_dataset_panel(documents_df, source_label)
    render_indexing_panel(documents_df, show_progress, pace=pace)
    render_search_panel(documents_df, show_progress, pace=pace)
    render_results_panel()


# ============================================================
# REMAINING SECTIONS
# ============================================================

def render_references():
    render_html('<div class="content-heading">References</div>')

    references = [
        "Sentence Transformers documentation",
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "Attention Is All You Need, the original transformer paper",
        "Scikit-learn cosine similarity documentation",
        "Introduction to Information Retrieval concepts",
        "Dense Passage Retrieval and vector search literature",
    ]

    for index, reference in enumerate(references, start=1):
        st.markdown(f"{index}. {reference}")

    render_html('<div class="content-subheading">Software Used</div>')

    for item in [
        "Python",
        "Streamlit",
        "SentenceTransformers",
        "NumPy",
        "Pandas",
        "Scikit-learn",
        "Plotly",
        "FPDF2",
        "Visual Studio Code",
    ]:
        st.markdown(f"- {item}")


def render_contributors():
    render_html('<div class="content-heading">Contributors</div>')

    st.write(
        """
        This virtual laboratory experiment was developed as an academic
        demonstration of dense embedding-based semantic search.
        """
    )

    st.markdown(
        """
**Experiment Developer:** Student Project Team

**Subject:** Natural Language Processing

**Technology:** Python, Streamlit and SentenceTransformers
"""
    )


def render_feedback():
    render_html('<div class="content-heading">Feedback</div>')
    st.write("Please provide your feedback about this virtual laboratory.")

    with st.form("feedback_form"):
        name = st.text_input("Name")
        rating = st.slider("Rate the experiment", 1, 5, 5)
        difficulty = st.selectbox("Experiment difficulty", ["Easy", "Moderate", "Difficult"])
        comments = st.text_area("Comments")
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        st.success("Thank you for your feedback.")
        st.write(
            {"Name": name, "Rating": rating, "Difficulty": difficulty, "Comments": comments}
        )


def render_report_generation():
    render_html('<div class="content-heading">Report Generation</div>')

    st.write(
        """
        Enter your details and generate a PDF report containing the aim, theory,
        experimental setup, quiz scores, observations and recorded trials.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input("Student Name")
        roll_number = st.text_input("Roll Number")

    with col2:
        department = st.text_input("Department", value="Computer Engineering")
        experiment_date = st.date_input("Experiment Date", value=datetime.now().date())

    observations = st.text_area(
        "Observations",
        value=(
            "The document collection was preprocessed and converted into dense "
            "embeddings batch by batch, and the vectors were stored as an embedding "
            "index. Each query was encoded with the same model and compared against "
            "the index using cosine similarity. The Top-K limit and the similarity "
            "threshold controlled how many documents were returned."
        ),
        height=150,
    )

    if st.session_state.trials:
        render_html('<div class="content-subheading">Recorded Trials</div>')

        trials_df = pd.DataFrame(st.session_state.trials)
        st.dataframe(trials_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Trial CSV",
            data=trials_df.to_csv(index=False),
            file_name="semantic_search_trials.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No trials recorded yet. Perform a search and record a trial.")

    if st.button("Generate PDF Report", use_container_width=True):
        if not student_name.strip():
            st.warning("Please enter student name.")

        elif not roll_number.strip():
            st.warning("Please enter roll number.")

        else:
            pdf_bytes = generate_pdf_report(
                student_name=student_name,
                roll_number=roll_number,
                department=department,
                experiment_date=experiment_date.strftime("%d-%m-%Y"),
                observations=observations,
            )

            st.success("PDF report generated successfully.")

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="semantic_search_virtual_lab_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ============================================================
# ROUTER, FOOTER AND MAIN
# ============================================================

SECTION_RENDERERS = {
    "Aim and Introduction": render_aim_and_introduction,
    "Theory and Application": render_theory_and_application,
    "Pretest": render_pretest,
    "Procedure": render_procedure,
    "Simulation": render_simulation,
    "Posttest": render_posttest,
    "References": render_references,
    "Contributors": render_contributors,
    "Feedback": render_feedback,
    "Report Generation": render_report_generation,
}


def render_selected_content():
    renderer = SECTION_RENDERERS.get(
        st.session_state.active_section, render_aim_and_introduction
    )
    renderer()


def render_footer():
    render_html(
        """
        <div class="vlab-footer">
            <div>Community Links</div>
            <div>Contact Us</div>
            <div>Follow Us</div>
        </div>
        """
    )


def main():
    render_top_header()
    render_breadcrumb()

    left_column, main_column = st.columns([1.15, 5], gap="small")

    with left_column:
        with st.container(key="left_nav_panel"):
            render_left_navigation()

    with main_column:
        with st.container(key="main_content_panel"):
            render_html(f'<div class="experiment-heading">{EXPERIMENT_TITLE}</div>')
            render_selected_content()

    render_footer()


if __name__ == "__main__":
    main()
