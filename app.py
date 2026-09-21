import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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
        "posttest_submitted": False,
        "posttest_score": 0,
        "posttest_attempted": 0,
        "pretest_page": 1,
        "posttest_page": 1,
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
        f"Pretest Score: {st.session_state.pretest_score}/{len(PRETEST_QUESTIONS)}",
        f"Posttest Score: {st.session_state.posttest_score}/{len(POSTTEST_QUESTIONS)}",
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
    render_html('<div class="content-heading">Theory and Application</div>')

    render_html('<div class="content-subheading">Dense Embeddings</div>')
    st.write(THEORY_EMBEDDINGS)

    st.write(
        """
        The model used here, all-MiniLM-L6-v2, is a six-layer transformer that
        produces a 384-dimensional vector for any input text. Tokens are first
        embedded and contextualised through self-attention, and the token vectors
        are then mean pooled into a single sentence vector. The model was trained
        with a siamese objective on sentence pairs, so that pairs with the same
        meaning are pulled together in the vector space and unrelated pairs are
        pushed apart.
        """
    )

    render_html('<div class="content-subheading">Cosine Similarity</div>')
    st.write(THEORY_COSINE)

    st.latex(r"\text{Cosine Similarity}(A,B) = \frac{A \cdot B}{\|A\|\|B\|}")

    st.write(
        """
        Because every embedding produced in this experiment is normalised to unit
        length, the denominator becomes one and the similarity reduces to a plain
        dot product. Comparing a query against the whole collection is therefore a
        single matrix multiplication, which is why the search takes only a few
        milliseconds for a few hundred documents.
        """
    )

    render_html('<div class="content-subheading">The Two-Stage Retrieval Pipeline</div>')

    st.markdown("**Stage A — Document Indexing (performed once)**")
    render_pipeline(INDEX_STAGES)

    st.markdown("**Stage B — Query Search (performed per query)**")
    render_pipeline(QUERY_STAGES)

    render_html('<div class="content-subheading">Important Terms</div>')

    terms = pd.DataFrame(
        [
            {"Term": "Embedding", "Meaning": "Numerical vector representation of text."},
            {"Term": "Dense Vector", "Meaning": "Fixed-length vector whose values are mostly non-zero."},
            {"Term": "Transformer", "Meaning": "Neural architecture based on self-attention."},
            {"Term": "Mean Pooling", "Meaning": "Averaging token vectors into one sentence vector."},
            {"Term": "Embedding Index", "Meaning": "Stored matrix of document embeddings reused for every query."},
            {"Term": "Cosine Similarity", "Meaning": "Similarity based on the angle between two vectors."},
            {"Term": "Top-K", "Meaning": "Number of highest ranked documents returned."},
            {"Term": "Similarity Threshold", "Meaning": "Minimum score a document must reach to be shown."},
            {"Term": "Vector Database", "Meaning": "System that stores embeddings and serves similarity search."},
            {"Term": "Semantic Search", "Meaning": "Retrieval based on meaning instead of exact keywords."},
        ]
    )

    st.dataframe(terms, use_container_width=True, hide_index=True)

    render_html('<div class="content-heading">Application</div>')

    st.write(
        """
        Dense embedding-based retrieval is the backbone of most modern search and
        assistant systems. The areas below describe where it is used, what problem
        it solves in that setting, and why a purely keyword based system is not
        sufficient.
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
                    <div class="app-card">
                        <div class="app-card-title">{escape_html(title)}</div>
                        <div class="app-card-text">{escape_html(description)}</div>
                    </div>
                    """
                )

    render_html('<div class="content-subheading">Practical Limitations</div>')

    for limitation in APPLICATION_LIMITATIONS:
        st.markdown(f"- {limitation}")

    render_html(
        """
        <div class="info-box">
            In production, dense retrieval is usually combined with a lexical
            ranker such as BM25 and followed by a cross-encoder re-ranker. The
            dense stage supplies recall, the lexical stage protects exact matches,
            and the re-ranker sharpens the final order.
        </div>
        """
    )


# ============================================================
# QUIZ RENDERING (50 questions per test, 10 per page)
# ============================================================

QUESTIONS_PER_PAGE = 10


def render_quiz(bank, prefix, heading, intro):
    render_html(f'<div class="content-heading">{heading}</div>')
    st.write(intro)

    total = len(bank)
    pages = (total + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE

    answered = sum(
        1 for index in range(total) if st.session_state.get(f"{prefix}_answer_{index}")
    )

    render_html(
        f'<div class="quiz-progress">Questions attempted: '
        f"{answered} of {total}</div>"
    )

    st.progress(answered / total)

    page = st.selectbox(
        "Question set",
        list(range(1, pages + 1)),
        index=st.session_state[f"{prefix}_page"] - 1,
        format_func=lambda p: (
            f"Questions {(p - 1) * QUESTIONS_PER_PAGE + 1}"
            f"-{min(p * QUESTIONS_PER_PAGE, total)}"
        ),
        key=f"{prefix}_page_select",
    )

    st.session_state[f"{prefix}_page"] = page

    start = (page - 1) * QUESTIONS_PER_PAGE
    end = min(start + QUESTIONS_PER_PAGE, total)

    for index in range(start, end):
        question = bank[index]

        options = question["options"]
        shift = index % len(options)
        ordered = options[shift:] + options[:shift]

        st.markdown(f"**Q{index + 1}. {question['question']}**")

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

    action_col, reset_col = st.columns(2)

    with action_col:
        if st.button("Submit and Evaluate", key=f"{prefix}_submit", use_container_width=True):
            score = 0
            attempted = 0

            for index, question in enumerate(bank):
                chosen = st.session_state.get(f"{prefix}_answer_{index}")

                if chosen is not None:
                    attempted += 1

                    if chosen == question["answer"]:
                        score += 1

            st.session_state[f"{prefix}_score"] = score
            st.session_state[f"{prefix}_attempted"] = attempted
            st.session_state[f"{prefix}_submitted"] = True

    with reset_col:
        if st.button("Clear All Answers", key=f"{prefix}_reset", use_container_width=True):
            for index in range(total):
                st.session_state.pop(f"{prefix}_answer_{index}", None)

            st.session_state[f"{prefix}_submitted"] = False
            st.session_state[f"{prefix}_score"] = 0
            st.session_state[f"{prefix}_attempted"] = 0

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
        "This pretest contains 50 questions covering the basic concepts needed "
        "before performing the experiment. Questions are shown ten at a time; "
        "your answers are retained while you move between sets.",
    )


def render_posttest():
    render_quiz(
        POSTTEST_QUESTIONS,
        "posttest",
        "Posttest",
        "This posttest contains 50 questions on indexing, similarity, ranking, "
        "evaluation and the practical behaviour you observed in the simulation. "
        "Attempt all sets and submit to see your score with the correct answers.",
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

def run_live_indexing(documents_df, live, batch_size=16, pause=0.15):
    """Encode the collection batch by batch, showing every step as it happens."""
    model_placeholder = st.empty()
    pipeline_placeholder = st.empty()
    log_placeholder = st.empty()
    progress_placeholder = st.empty()
    metric_placeholder = st.empty()
    preview_placeholder = st.empty()

    logs = []

    def log(message, style="log-run"):
        logs.append(f'<span class="{style}">{message}</span>')
        if live:
            log_placeholder.markdown(log_markup(logs[-14:]), unsafe_allow_html=True)

    def stage(active):
        if live:
            pipeline_placeholder.markdown(
                pipeline_markup(INDEX_STAGES, active), unsafe_allow_html=True
            )

    start = time.perf_counter()

    # ---- Stage 0: documents ----
    stage(0)
    log(f"[1/5] Reading document collection ... {len(documents_df)} documents loaded.")
    if live:
        time.sleep(pause)

    # ---- Stage 1: preprocessing ----
    stage(1)
    texts = documents_df["content"].astype(str).tolist()
    total_words = int(documents_df["word_count"].sum())
    log(f"[2/5] Preprocessing text ... {total_words} words, {len(texts)} passages queued.")

    if live:
        sample = pd.DataFrame(
            {
                "id": documents_df["id"].head(5),
                "characters": [len(t) for t in texts[:5]],
                "words": [len(t.split()) for t in texts[:5]],
                "first tokens": [" | ".join(t.split()[:8]) + " ..." for t in texts[:5]],
            }
        )
        preview_placeholder.dataframe(sample, use_container_width=True, hide_index=True)
        time.sleep(pause)

    # ---- Stage 2: model ----
    stage(2)
    log(f"[3/5] Loading SentenceTransformer model '{MODEL_NAME}' ...")
    if live:
        model_placeholder.caption("Loading the transformer (cached after the first run).")

    model = load_embedding_model()
    log("      Model ready. Tokenizer and 6 transformer layers initialised.", "log-ok")
    if live:
        model_placeholder.empty()

    # ---- Stage 3: encoding ----
    stage(3)
    log(f"[4/5] Encoding documents in batches of {batch_size} ...")

    progress = progress_placeholder.progress(0.0) if live else None

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

        if live:
            progress.progress(encoded / len(texts))

            log(
                f"      batch {len(chunks):>3} | documents {encoded}/{len(texts)}"
                f" | vector shape {vectors.shape}"
            )

            running = np.vstack(chunks)

            metric_placeholder.markdown(
                f"**Encoded:** {encoded}/{len(texts)} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Index shape:** {running.shape} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Mean vector norm:** {np.linalg.norm(running, axis=1).mean():.3f}"
            )

            preview_placeholder.dataframe(
                pd.DataFrame(
                    vectors[: min(3, len(vectors)), :12].round(4),
                    index=documents_df["id"].iloc[start_index:start_index + min(3, len(vectors))],
                    columns=[f"d{i}" for i in range(12)],
                ),
                use_container_width=True,
            )

            time.sleep(pause / 2)

    embeddings = np.vstack(chunks)

    # ---- Stage 4: index ----
    stage(4)
    elapsed = time.perf_counter() - start
    log(
        f"[5/5] Embedding index built: {embeddings.shape[0]} vectors x "
        f"{embeddings.shape[1]} dimensions in {elapsed:.2f} s.",
        "log-ok",
    )

    if live:
        pipeline_placeholder.markdown(
            pipeline_markup(INDEX_STAGES, len(INDEX_STAGES)), unsafe_allow_html=True
        )
        progress_placeholder.empty()

    store_index(embeddings, documents_df, elapsed)

    return embeddings


def render_embedding_map(embeddings, documents_df):
    points = pca_projection(embeddings)

    frame = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "Title": documents_df["title"],
            "Category": documents_df["category"],
        }
    )

    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="Category",
        hover_name="Title",
        title="Embedding space (PCA projection of the index to 2 dimensions)",
    )

    figure.update_layout(height=470, xaxis_title="Component 1", yaxis_title="Component 2")
    figure.update_traces(marker={"size": 9, "opacity": 0.8})

    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Each point is one document vector. Documents from the same subject area "
        "form clusters, which is the property cosine similarity exploits at query time."
    )


def render_indexing_panel(documents_df, live):
    render_html('<div class="stage-label">STAGE A</div>')
    render_html('<div class="content-subheading">2. Document Indexing (live)</div>')

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
        run_live_indexing(documents_df, live)
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

        with st.expander("View the embedding space map", expanded=False):
            render_embedding_map(embeddings, st.session_state.indexed_documents)


# ============================================================
# SIMULATION: STAGE B WITH LIVE VISUALISATION
# ============================================================


def render_mock_semantic_animation():
    """Beginner-friendly visual demonstration using illustrative, non-live data."""
    render_html(
        """
        <div class="content-subheading">Interactive Mock Animation</div>
        <div class="info-box">
            This animation uses example documents and illustrative scores.
            It is designed to explain the complete workflow visually before running
            the actual semantic search on the selected dataset.
        </div>
        """
    )

    mock_query = "How do computers learn from data?"
    mock_documents = [
        ("Document A", "Introduction to Machine Learning", 0.94, "Very relevant"),
        ("Document B", "Supervised Learning Algorithms", 0.88, "Relevant"),
        ("Document C", "Computer Networks", 0.31, "Weak match"),
        ("Document D", "Database Management", 0.22, "Low match"),
    ]

    if st.button("▶ Start Mock Animation", key="start_mock_animation",
                 use_container_width=True):
        stage_box = st.empty()
        visual_box = st.empty()
        explanation_box = st.empty()
        progress = st.progress(0)

        def show_stage(number, title, description, percent):
            stage_box.markdown(
                f"""
                <div class="app-card">
                    <div class="app-card-title">Step {number}: {title}</div>
                    <div class="app-card-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress.progress(percent)

        def show_flow(left, middle, right, color="#2696d2"):
            visual_box.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:center;
                            gap:10px;flex-wrap:wrap;margin:14px 0;">
                    <div style="flex:1;min-width:145px;text-align:center;
                                padding:18px 10px;border:2px solid {color};
                                border-radius:12px;background:#f5fbff;
                                font-weight:700;color:#245b7c;">{left}</div>
                    <div style="font-size:28px;color:#f47721;font-weight:700;">→</div>
                    <div style="flex:1;min-width:145px;text-align:center;
                                padding:18px 10px;border:2px solid {color};
                                border-radius:12px;background:#f5fbff;
                                font-weight:700;color:#245b7c;">{middle}</div>
                    <div style="font-size:28px;color:#f47721;font-weight:700;">→</div>
                    <div style="flex:1;min-width:145px;text-align:center;
                                padding:18px 10px;border:2px solid {color};
                                border-radius:12px;background:#f5fbff;
                                font-weight:700;color:#245b7c;">{right}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        show_stage(1, "Documents enter the system",
                   "The system receives a collection of documents. Each document contains text and a topic.",
                   12)
        show_flow("📄 Document 1", "📄 Document 2", "📄 Document 3")
        explanation_box.info("Think of this as placing books on a table before organizing them.")
        time.sleep(1.2)

        show_stage(2, "The AI model reads the text",
                   "The SentenceTransformer model processes the meaning of every document.",
                   28)
        show_flow("Text", "🧠 AI Model", "Meaning")
        explanation_box.info("The model does not simply count matching words. It converts meaning into numbers.")
        time.sleep(1.2)

        show_stage(3, "Text becomes a vector",
                   "Every document is represented by a dense numerical vector and stored in the index.",
                   45)
        show_flow("Document meaning", "🔢 Vector", "🗃️ Vector index")
        explanation_box.info("A vector is a list of numbers. Similar meanings are represented by vectors that point in similar directions.")
        time.sleep(1.2)

        show_stage(4, "The user enters a query",
                   f'Example query: "{mock_query}"',
                   58)
        show_flow("User question", "🧠 Same AI Model", "Query vector", "#f47721")
        explanation_box.info("The query is converted using the same model so that it can be compared with document vectors.")
        time.sleep(1.2)

        show_stage(5, "Similarity is calculated",
                   "The query vector is compared with every document vector using cosine similarity.",
                   73)
        visual_box.markdown(
            """
            <div style="border:1px solid #d8e7f0;border-radius:12px;padding:18px;
                        background:#fbfdff;">
                <div style="font-weight:700;color:#245b7c;margin-bottom:12px;">
                    Query vector compared with stored vectors
                </div>
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="width:120px;">Document A</span>
                        <div style="height:18px;flex:1;background:#d9f2df;border-radius:20px;">
                            <div style="width:94%;height:18px;background:#35a853;border-radius:20px;"></div>
                        </div><b>0.94</b>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="width:120px;">Document B</span>
                        <div style="height:18px;flex:1;background:#d9f2df;border-radius:20px;">
                            <div style="width:88%;height:18px;background:#61b875;border-radius:20px;"></div>
                        </div><b>0.88</b>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="width:120px;">Document C</span>
                        <div style="height:18px;flex:1;background:#f8ead9;border-radius:20px;">
                            <div style="width:31%;height:18px;background:#e6a04e;border-radius:20px;"></div>
                        </div><b>0.31</b>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="width:120px;">Document D</span>
                        <div style="height:18px;flex:1;background:#f8ead9;border-radius:20px;">
                            <div style="width:22%;height:18px;background:#e6a04e;border-radius:20px;"></div>
                        </div><b>0.22</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        explanation_box.info("Higher scores mean the document is more semantically related to the query.")
        time.sleep(1.5)

        show_stage(6, "Documents are ranked and filtered",
                   "The documents are sorted from the highest similarity score to the lowest score.",
                   88)
        rows = "".join(
            f"""
            <tr>
                <td style="padding:9px;border-bottom:1px solid #e5e5e5;">{i}</td>
                <td style="padding:9px;border-bottom:1px solid #e5e5e5;">{title}</td>
                <td style="padding:9px;border-bottom:1px solid #e5e5e5;font-weight:700;">{score:.2f}</td>
                <td style="padding:9px;border-bottom:1px solid #e5e5e5;">{label}</td>
            </tr>
            """
            for i, (_, title, score, label) in enumerate(mock_documents, start=1)
        )
        visual_box.markdown(
            f"""
            <div style="border:1px solid #d8e7f0;border-radius:12px;padding:12px;background:#fff;">
                <div style="font-weight:700;color:#245b7c;margin-bottom:10px;">Ranked results</div>
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:#f5fbff;">
                        <th style="padding:9px;text-align:left;">Rank</th>
                        <th style="padding:9px;text-align:left;">Document</th>
                        <th style="padding:9px;text-align:left;">Score</th>
                        <th style="padding:9px;text-align:left;">Interpretation</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        explanation_box.info("Top-K keeps only the requested number of results. The similarity threshold can remove weak matches.")
        time.sleep(1.2)

        show_stage(7, "Final results are shown to the user",
                   "The most relevant documents are displayed with their similarity scores.",
                   100)
        show_flow("Query", "Ranked documents", "Useful answer", "#35a853")
        explanation_box.success("Mock animation completed. You can now run the actual search below using your real dataset.")

def run_live_search(query, documents_df, embeddings, top_k, threshold, live, pause=0.35):
    pipeline_placeholder = st.empty()
    log_placeholder = st.empty()
    detail_placeholder = st.container()

    logs = []

    def log(message, style="log-run"):
        logs.append(f'<span class="{style}">{message}</span>')
        if live:
            log_placeholder.markdown(log_markup(logs[-12:]), unsafe_allow_html=True)

    def stage(active):
        if live:
            pipeline_placeholder.markdown(
                pipeline_markup(QUERY_STAGES, active), unsafe_allow_html=True
            )

    start = time.perf_counter()

    # ---- 0: query received ----
    stage(0)
    log(f"[1/6] Query received: \"{escape_html(query)}\"")
    if live:
        time.sleep(pause)

    # ---- 1: preprocessing ----
    stage(1)
    tokens = query.split()
    log(f"[2/6] Preprocessing: {len(query)} characters, {len(tokens)} whitespace tokens.")
    if live:
        time.sleep(pause)

    # ---- 2: query embedding ----
    stage(2)
    log("[3/6] Encoding the query with the same model used for the documents ...")

    query_vector, scores = compute_similarities(query, documents_df, embeddings)

    log(
        f"      Query vector generated: shape ({query_vector.shape[0]},), "
        f"norm {np.linalg.norm(query_vector):.3f}.",
        "log-ok",
    )

    if live:
        with detail_placeholder:
            preview = pd.DataFrame(
                {
                    "dimension": [f"d{i}" for i in range(24)],
                    "value": query_vector[:24],
                }
            )

            vector_figure = px.bar(
                preview,
                x="dimension",
                y="value",
                title="Query embedding (first 24 of "
                f"{query_vector.shape[0]} dimensions)",
            )
            vector_figure.update_layout(height=260, xaxis_title="", yaxis_title="value")

            st.plotly_chart(vector_figure, use_container_width=True)

        time.sleep(pause)

    # ---- 3: cosine similarity ----
    stage(3)
    log(
        f"[4/6] Computing cosine similarity against {len(documents_df)} document vectors "
        "(single matrix multiplication of unit vectors)."
    )

    if live:
        with detail_placeholder:
            histogram = px.histogram(
                pd.DataFrame({"Cosine Similarity": scores}),
                x="Cosine Similarity",
                nbins=40,
                title="Distribution of similarity scores across the whole collection",
            )

            histogram.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="#f47721",
                annotation_text=f"threshold {threshold:.2f}",
            )

            histogram.update_layout(height=300, yaxis_title="Number of documents")

            st.plotly_chart(histogram, use_container_width=True)

        log(
            f"      max {scores.max():.4f} | mean {scores.mean():.4f} | "
            f"min {scores.min():.4f}"
        )
        time.sleep(pause)

    # ---- 4: ranking ----
    stage(4)
    above = int((scores >= threshold).sum())
    log(
        f"[5/6] Ranking documents by score. {above} of {len(scores)} documents "
        f"reach the threshold of {threshold:.2f}."
    )
    if live:
        time.sleep(pause)

    # ---- 5: top-k ----
    stage(5)
    results = assemble_results(scores, documents_df, top_k, threshold)
    elapsed = time.perf_counter() - start

    log(
        f"[6/6] Returning the Top-{top_k} documents: {len(results)} results "
        f"in {elapsed * 1000:.1f} ms.",
        "log-ok",
    )

    if live:
        pipeline_placeholder.markdown(
            pipeline_markup(QUERY_STAGES, len(QUERY_STAGES)), unsafe_allow_html=True
        )

    return results, elapsed, int(query_vector.shape[0]), scores


def render_search_panel(documents_df, live):
    render_html('<div class="stage-label">STAGE B</div>')
    render_html('<div class="content-subheading">3. Query Search (live)</div>')

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
                live,
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

    render_html(
        """
        <div class="info-box">
            First understand the complete semantic-search workflow through the
            visual mock animation. After that, use the actual dataset simulation
            to build the index and perform a real search.
        </div>
        """
    )

    render_mock_semantic_animation()

    render_html('<div class="content-heading">Actual Dataset Simulation</div>')

    live = st.checkbox(
        "Show technical live details during the actual search",
        value=True,
        key="actual_live_details",
    )

    documents_df = get_active_documents()
    _, source_label = load_base_documents()

    if st.session_state.uploaded_documents is not None:
        source_label = f"{source_label} + uploaded file"

    render_dataset_panel(documents_df, source_label)
    render_indexing_panel(documents_df, live)
    render_search_panel(documents_df, live)
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
