TrojnLLM

Work flow:
1. Calculate Q-value
2. Add some perturbation

Ideas:
Classifier Weights Manipulation: For tasks like sentiment analysis or classification, the final layers of BERT are fine-tuned to perform these specific tasks. Poisoning could result in significant changes to the weights in these layers, causing the model to misclassify inputs in a way that serves the attacker's objectives.

Increased Vulnerability to Adversarial Attacks: The changes in parameters might not only serve the immediate goals of the poisoning attack (e.g., biasing the model towards a certain output) but also make the model more susceptible to future adversarial attacks by introducing vulnerabilities or reducing the model's overall robustness.