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

    return df[["id", "title", "category", "content", "source", "word_count"]].reset_index(drop=True)


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
    "Understand dense text embeddings.",
    "Generate vector representations using SentenceTransformer.",
    "Understand cosine similarity between vectors.",
    "Build an embedding index over a document collection.",
    "Retrieve semantically similar documents.",
    "Analyze the effect of Top-K and the similarity threshold.",
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
    "Build the embedding index for all documents (Stage A).",
    "Inspect the generated embeddings and index statistics.",
    "Enter a natural language query (Stage B).",
    "Generate the query embedding.",
    "Calculate cosine similarity between the query and every document.",
    "Apply the similarity threshold and keep the Top-K documents.",
    "Analyse the ranked results and generate a PDF report.",
]

QUIZ_QUESTIONS = [
    {
        "question": "What is the main purpose of a sentence embedding?",
        "options": [
            "To convert text into a numerical vector",
            "To delete all words from a sentence",
            "To convert text into an image",
            "To sort documents alphabetically",
        ],
        "answer": "To convert text into a numerical vector",
    },
    {
        "question": "Which similarity measure is used in this experiment?",
        "options": [
            "Euclidean distance only",
            "Cosine similarity",
            "Manhattan distance only",
            "Jaccard index only",
        ],
        "answer": "Cosine similarity",
    },
    {
        "question": "What does Top-K represent?",
        "options": [
            "The number of input characters",
            "The number of model layers",
            "The number of returned documents",
            "The embedding dimension",
        ],
        "answer": "The number of returned documents",
    },
    {
        "question": "Which model is used to generate embeddings?",
        "options": [
            "SentenceTransformer",
            "Linear Regression",
            "Decision Tree",
            "K-Means only",
        ],
        "answer": "SentenceTransformer",
    },
    {
        "question": "A cosine similarity score close to 1 generally indicates:",
        "options": [
            "High semantic similarity",
            "No relationship",
            "A completely empty document",
            "A failed search",
        ],
        "answer": "High semantic similarity",
    },
    {
        "question": "What is the purpose of the embedding index?",
        "options": [
            "To store document embeddings so every query reuses them",
            "To delete unused documents",
            "To translate documents into other languages",
            "To compress images inside documents",
        ],
        "answer": "To store document embeddings so every query reuses them",
    },
    {
        "question": "Raising the minimum similarity threshold will usually:",
        "options": [
            "Return fewer but more relevant documents",
            "Return every document in the collection",
            "Increase the embedding dimension",
            "Change the model architecture",
        ],
        "answer": "Return fewer but more relevant documents",
    },
]


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    defaults = {
        "active_section": "Aim",
        "last_query": "",
        "last_results": [],
        "last_query_time": 0.0,
        "last_embedding_dimension": 0,
        "last_documents_compared": 0,
        "last_threshold": 0.25,
        "trials": [],
        "quiz_submitted": False,
        "quiz_score": 0,
        "index_built": False,
        "document_embeddings": None,
        "indexed_documents": None,
        "index_signature": "",
        "index_build_time": 0.0,
        "uploaded_documents": None,
        "upload_message": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# MODEL AND INDEXING
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def build_embedding_index(documents_df):
    model = load_embedding_model()

    start = time.perf_counter()

    embeddings = model.encode(
        documents_df["content"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    st.session_state.document_embeddings = embeddings
    st.session_state.indexed_documents = documents_df.reset_index(drop=True)
    st.session_state.index_signature = dataset_signature(documents_df)
    st.session_state.index_build_time = elapsed
    st.session_state.index_built = True

    return embeddings, elapsed


def index_is_current(documents_df):
    return (
        st.session_state.index_built
        and st.session_state.document_embeddings is not None
        and st.session_state.index_signature == dataset_signature(documents_df)
    )


def perform_semantic_search(query, top_k, threshold):
    """Stage B: query embedding -> cosine similarity -> threshold -> Top-K."""
    if not query or not query.strip():
        return [], 0.0, 0, 0

    model = load_embedding_model()
    documents_df = st.session_state.indexed_documents
    document_embeddings = st.session_state.document_embeddings

    start = time.perf_counter()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = cosine_similarity(query_embedding, document_embeddings)[0]

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

    query_time = time.perf_counter() - start

    return results, query_time, int(query_embedding.shape[1]), len(documents_df)


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

    setup_lines = [
        f"Embedding Model: {MODEL_NAME}",
        f"Documents in Collection: {len(documents_df)}",
        f"Categories: {documents_df['category'].nunique()}",
        f"Embedding Dimension: {st.session_state.last_embedding_dimension or 384}",
        f"Index Status: {'Ready' if st.session_state.index_built else 'Not built'}",
    ]

    for line in setup_lines:
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
            KGIRS VIRTUAL LAB
            &nbsp;&rsaquo;&nbsp;
            Experiment
        </div>
        """
    )


def render_pipeline(stages):
    parts = []
    for position, stage in enumerate(stages):
        if position:
            parts.append('<span class="pipeline-arrow">&rarr;</span>')
        parts.append(f'<span class="pipeline-stage">{escape_html(stage)}</span>')

    render_html('<div class="pipeline">' + "".join(parts) + "</div>")


def render_left_navigation():
    render_html('<div class="left-navigation-title">Experiment Sections</div>')

    sections = [
        "Aim",
        "Introduction",
        "Theory",
        "Pretest",
        "Procedure",
        "Simulation",
        "Application",
        "Posttest",
        "References",
        "Contributors",
        "Feedback",
        "Report Generation",
    ]

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
# STATIC SECTIONS
# ============================================================

def render_aim():
    render_html('<div class="content-heading">Aim of the experiment</div>')

    render_html(
        """
        <div class="aim-list">
            <ul>
                <li>To implement semantic search using dense embeddings.</li>
                <li>To convert a document collection and a query into numerical vectors.</li>
                <li>To build and inspect an embedding index.</li>
                <li>To calculate cosine similarity between embeddings.</li>
                <li>To retrieve the most semantically relevant documents.</li>
                <li>To study the effect of the Top-K parameter and the similarity threshold.</li>
            </ul>
        </div>
        """
    )


def render_introduction():
    render_html('<div class="content-heading">Introduction</div>')

    render_html(
        f'<div class="content-text">{to_html_paragraphs(THEORY_INTRODUCTION)}</div>'
    )

    render_html('<div class="content-subheading">Why Semantic Search?</div>')

    st.write(
        """
        Keyword search depends mainly on matching exact words. Semantic
        search understands the meaning of the query and can retrieve
        related documents even when the wording is different.
        """
    )


def render_theory():
    render_html('<div class="content-heading">Theory</div>')

    render_html('<div class="content-subheading">Dense Embeddings</div>')
    st.write(THEORY_EMBEDDINGS)

    render_html('<div class="content-subheading">Cosine Similarity</div>')
    st.write(THEORY_COSINE)

    st.latex(r"\text{Cosine Similarity}(A,B) = \frac{A \cdot B}{\|A\|\|B\|}")

    render_html('<div class="content-subheading">Retrieval Pipeline</div>')

    st.markdown("**Stage A — Document Indexing**")
    render_pipeline(
        ["Documents", "Preprocessing", "SentenceTransformer", "Dense Embeddings", "Embedding Index"]
    )

    st.markdown("**Stage B — Query Search**")
    render_pipeline(
        ["User Query", "Query Embedding", "Cosine Similarity", "Ranking", "Top-K Documents"]
    )

    render_html('<div class="content-subheading">Important Terms</div>')

    terms = pd.DataFrame(
        [
            {"Term": "Embedding", "Meaning": "Numerical vector representation of text."},
            {"Term": "Dense Vector", "Meaning": "Vector containing numerical values in many dimensions."},
            {"Term": "Transformer", "Meaning": "Neural network architecture used for language understanding."},
            {"Term": "SentenceTransformer", "Meaning": "Model that converts sentences into meaningful vectors."},
            {"Term": "Embedding Index", "Meaning": "Stored matrix of document embeddings reused for every query."},
            {"Term": "Cosine Similarity", "Meaning": "Similarity measure based on the angle between vectors."},
            {"Term": "Top-K", "Meaning": "Number of highest-ranked documents returned."},
            {"Term": "Similarity Threshold", "Meaning": "Minimum score a document must reach to be shown."},
            {"Term": "Semantic Search", "Meaning": "Search based on meaning instead of exact keywords."},
        ]
    )

    st.dataframe(terms, use_container_width=True, hide_index=True)


def render_pretest():
    render_html('<div class="content-heading">Pretest</div>')
    st.write("Answer the following questions before performing the experiment.")

    with st.form("pretest_form"):
        answers = {}

        for index, question in enumerate(QUIZ_QUESTIONS[:3]):
            st.markdown(f"**Q{index + 1}. {question['question']}**")
            answers[index] = st.radio("Select an answer:", question["options"], key=f"pretest_{index}")

        submitted = st.form_submit_button("Submit Pretest")

    if submitted:
        score = sum(
            1
            for index, question in enumerate(QUIZ_QUESTIONS[:3])
            if answers[index] == question["answer"]
        )
        st.success(f"Your pretest score is {score}/3.")


def render_procedure():
    render_html('<div class="content-heading">Procedure</div>')

    for index, step in enumerate(PROCEDURE_STEPS, start=1):
        render_html(f'<div class="content-text"><b>{index}.</b> {escape_html(step)}</div>')

    render_html('<div class="content-subheading">Input</div>')
    st.write("A document collection and a natural language query entered by the user.")

    render_html('<div class="content-subheading">Output</div>')
    st.write("Ranked documents with cosine similarity scores, filtered by Top-K and the threshold.")


# ============================================================
# SIMULATION SECTION
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


def render_indexing_panel(documents_df):
    render_html('<div class="stage-label">STAGE A</div>')
    render_html('<div class="content-subheading">2. Document Indexing</div>')

    render_pipeline(
        ["Documents", "Preprocessing", "SentenceTransformer", "Dense Embeddings", "Embedding Index"]
    )

    if not index_is_current(documents_df):
        render_html(
            """
            <div class="info-box">
                The embedding index is not up to date with the current document
                collection. Build the index before running a search.
            </div>
            """
        )

    if st.button("Build Embedding Index", use_container_width=True):
        with st.spinner(f"Generating embeddings for {len(documents_df)} documents..."):
            build_embedding_index(documents_df)

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

            st.caption(
                "First 5 documents, first 16 of "
                f"{embeddings.shape[1]} dimensions."
            )

            preview = pd.DataFrame(
                embeddings[:5, :16].round(4),
                index=st.session_state.indexed_documents["id"].head(5),
                columns=[f"d{i}" for i in range(16)],
            )

            st.dataframe(preview, use_container_width=True)


def render_search_panel(documents_df):
    render_html('<div class="stage-label">STAGE B</div>')
    render_html('<div class="content-subheading">3. Query Search</div>')

    render_pipeline(
        ["User Query", "Query Embedding", "Cosine Similarity", "Ranking", "Top-K Documents"]
    )

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
            with st.spinner("Encoding the query and computing cosine similarity..."):
                results, query_time, dimension, compared = perform_semantic_search(
                    query, top_k, threshold
                )

            st.session_state.last_query = query
            st.session_state.last_results = results
            st.session_state.last_query_time = query_time
            st.session_state.last_embedding_dimension = dimension
            st.session_state.last_documents_compared = compared
            st.session_state.last_threshold = threshold

            if not results:
                st.warning(
                    "No document reached the similarity threshold. "
                    "Lower the threshold or rephrase the query."
                )


def render_results_panel():
    results = st.session_state.last_results

    if not results:
        return

    render_html('<div class="content-subheading">4. Query Processing Information</div>')

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
            The simulation runs in two stages. In Stage A the document collection
            is converted into dense embeddings and stored as an index. In Stage B
            a query is embedded and compared with that index using cosine similarity.
        </div>
        """
    )

    documents_df = get_active_documents()
    _, source_label = load_base_documents()

    if st.session_state.uploaded_documents is not None:
        source_label = f"{source_label} + uploaded file"

    render_dataset_panel(documents_df, source_label)
    render_indexing_panel(documents_df)
    render_search_panel(documents_df)
    render_results_panel()


# ============================================================
# REMAINING SECTIONS
# ============================================================

def render_application():
    render_html('<div class="content-heading">Application</div>')

    st.write(
        "Dense embedding-based semantic search can be used in many real-world applications."
    )

    applications = [
        "Search engines",
        "Question-answering systems",
        "Document retrieval",
        "Chatbots and retrieval augmented generation",
        "Recommendation systems",
        "Research paper search",
        "Customer support systems",
        "Legal document search",
        "Medical information retrieval",
        "Educational learning platforms",
    ]

    for application in applications:
        st.markdown(f"- {application}")


def render_posttest():
    render_html('<div class="content-heading">Posttest</div>')
    st.write("Answer all questions to test your understanding.")

    with st.form("posttest_form"):
        answers = {}

        for index, question in enumerate(QUIZ_QUESTIONS):
            st.markdown(f"**Q{index + 1}. {question['question']}**")
            answers[index] = st.radio("Select an option:", question["options"], key=f"posttest_{index}")

        submitted = st.form_submit_button("Submit Quiz", use_container_width=True)

    if submitted:
        st.session_state.quiz_score = sum(
            1
            for index, question in enumerate(QUIZ_QUESTIONS)
            if answers[index] == question["answer"]
        )
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        score = st.session_state.quiz_score
        total = len(QUIZ_QUESTIONS)

        render_html(
            f"""
            <div class="result-box">
                <b>Quiz Result</b><br>
                Score: {score}/{total}<br>
                Percentage: {score / total * 100:.2f}%
            </div>
            """
        )


def render_references():
    render_html('<div class="content-heading">References</div>')

    references = [
        "Sentence Transformers documentation",
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "Scikit-learn cosine similarity documentation",
        "Natural Language Processing textbooks",
        "Introduction to Information Retrieval concepts",
        "Dense Retrieval and Vector Search concepts",
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
        Enter your details and generate a PDF report containing the aim,
        theory, experimental setup, observations and recorded trials.
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
            "The document collection was converted into dense embeddings and stored "
            "as an embedding index. Each query was encoded with the same model and "
            "compared against the index using cosine similarity. The Top-K limit and "
            "the similarity threshold controlled how many documents were returned."
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
    "Aim": render_aim,
    "Introduction": render_introduction,
    "Theory": render_theory,
    "Pretest": render_pretest,
    "Procedure": render_procedure,
    "Simulation": render_simulation,
    "Application": render_application,
    "Posttest": render_posttest,
    "References": render_references,
    "Contributors": render_contributors,
    "Feedback": render_feedback,
    "Report Generation": render_report_generation,
}


def render_selected_content():
    renderer = SECTION_RENDERERS.get(st.session_state.active_section, render_aim)
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