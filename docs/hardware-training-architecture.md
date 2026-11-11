# AI Lab — entraînement sur plusieurs matériels

Réalisation au 9 novembre : profils, adaptateurs et worker sont implémentés ; seule une chaîne MLX sur Mac est qualifiée. Voir [STATUS](../STATUS.md) pour distinguer opérations livrées, dépendances absentes et méthodes futures.

Révision du 8 novembre 2026. **Exigence confirmée par Kais :** choisir une architecture permettant d'exploiter les méthodes et moteurs avancés sur RTX PRO 6000 Blackwell, DGX Spark et d'autres machines. Le Mac reste la première machine connectée ; il ne définit pas les limites du produit. Les profils décrits ici sont proposés/documentés, sans entraînement AI Lab validé sur ces matériels.

## Décision structurante

Séparer quatre dimensions :

1. **L'objectif et la méthode** : SFT, LoRA/QLoRA, fine-tuning complet, préentraînement continu, DPO/GRPO, distillation ou quantification/QAT.
2. **Le backend d'exécution** : MLX, Unsloth, Transformers/TRL/PEFT, NeMo AutoModel/Megatron Bridge, NeMo RL.
3. **Le profil matériel et logiciel** : architecture CPU/GPU, mémoire, précision, versions, kernels, nombre de GPU et topologie.
4. **Le lieu et le transport** : Mac local, serveur SSH, GPU loué, ordonnanceur de cluster.

Runpod est un fournisseur de ressources, pas un moteur d'entraînement. Une machine louée et une station possédée peuvent utiliser le même profil. À l'inverse, deux machines NVIDIA peuvent nécessiter des environnements différents.

Le noyau AI Lab doit sélectionner une **recette versionnée compatible** avec ces dimensions. Les optimisations avancées restent disponibles dans les adaptateurs : ne pas réduire tous les moteurs au plus petit ensemble commun de paramètres. Les options propres au backend sont typées, validées et enregistrées dans la recette.

## Profils matériels

| Profil proposé | Distinctions documentées | Conséquence pour AI Lab |
| --- | --- | --- |
| Apple Silicon / Metal | Voie MLX, mémoire unifiée, environnement macOS. | Worker natif ; aucune dépendance CUDA obligatoire dans le noyau. |
| NVIDIA RTX / RTX PRO | Profils par génération, GPU exact et mémoire. RTX PRO 6000 Blackwell : éditions Workstation, Max-Q et Server, 96 Go GDDR7 ECC ; compute capability 12.0. | Profil CUDA adapté ; distinguer les éditions et la topologie PCIe. Ni recette ni débit uniques pour toutes les RTX. |
| DGX Spark GB10 | CPU ARM64, GPU compute capability 12.1, configurations actuelles de mémoire unifiée 64 ou 128 Go. | Image et dépendances ARM64 ; RAM disponible détectée ; recette spécifique Spark. Une image x86 ne convient pas automatiquement. |
| NVIDIA datacenter | Par exemple Hopper H100/H200 et Blackwell B200/GB200 ; B200/GB200 sont compute capability 10.0. | Profils et kernels séparés, détection HBM/NVLink/NVSwitch et réseau selon machine. |
| Plusieurs GPU ou nœuds | Communication, mémoire par GPU, placement des processus et stockage deviennent déterminants. | Profil de topologie en plus du profil de chaque GPU ; pas de simple somme de VRAM. |

Sources matérielles : [RTX PRO 6000, éditions](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/), [compute capabilities NVIDIA](https://developer.nvidia.com/cuda/gpus), [DGX Spark, release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html).

Le nom Blackwell ne garantit pas qu'un kernel SM100 fonctionne sur SM120/121. Les caractéristiques mémoire et les TOPS FP4 d'inférence ne constituent pas des mesures de débit d'entraînement. Détecter architecture, mémoire libre et capacités par opération.

Pour plusieurs Spark, qualifier le réseau et les collectifs. NVIDIA indique explicitement l'absence de GPUDirect RDMA sur Spark : ne pas activer automatiquement GDRCopy ou `nvidia-peermem` à partir de la présence de ConnectX. RDMA réseau et accès direct à la mémoire GPU sont distincts. [Restriction NVIDIA](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/related/1).

D'autres familles, notamment AMD/ROCm ou TPU, peuvent être ajoutées via de nouveaux profils/adaptateurs. Leur prise en charge n'est pas acquise par l'existence du contrat commun ; aucune compatibilité de ces familles n'a été qualifiée dans cette passe.

## Moteurs à retenir

| Usage | Choix proposé | Limite à respecter |
| --- | --- | --- |
| Adaptation sur Apple Silicon | MLX-LM | Matrice modèles/opérations propre à MLX. |
| Fine-tuning accessible et optimisé sur NVIDIA | Unsloth, notamment LoRA/QLoRA | Partir des recettes RTX/Spark documentées ; vérifier modèle, précision et versions. |
| Route Hugging Face portable et personnalisable | Transformers + PEFT + TRL, lancement Accelerate | SFT/DPO/GRPO et full fine-tuning selon trainer/recette ; FSDP2 ou DeepSpeed ZeRO si sharding requis. |
| Spark avec recette NVIDIA | NeMo AutoModel ou PyTorch/Unsloth selon recette validée | ARM64 et réglages mémoire explicites ; ne pas appliquer une recette B200. |
| Gros entraînements distribués, modèles couverts et MoE | NeMo AutoModel/Megatron Bridge | Recettes spécialisées de préentraînement, SFT, LoRA ; parallélisme adapté au modèle et au réseau. |
| Post-training RL distribué | TRL pour commencer ; NeMo RL pour la voie cluster spécialisée | Génération, récompenses, modèle de référence éventuel et transfert des poids font partie du travail. |
| Quantification NVIDIA avancée | NVIDIA Model Optimizer ; TorchAO selon workflow qualifié | Distinguer PTQ, QAT, QAD, QLoRA et calcul d'entraînement en basse précision. |

Unsloth documente Blackwell et NVIDIA propose un playbook Spark. Son DDP réplique le modèle sur chaque GPU : il augmente le débit potentiel, pas la capacité mémoire d'une réplique. Pour répartir états/paramètres, retenir une recette FSDP/ZeRO spécifique ; `device_map` n'est pas un remplacement de l'entraînement distribué. [Unsloth Blackwell](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth), [Unsloth Spark](https://build.nvidia.com/spark/unsloth/instructions), [DDP](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp), [PEFT/FSDP](https://huggingface.co/docs/peft/accelerate/fsdp).

FSDP2 répartit les états d'entraînement ; Accelerate aide à lancer/configurer. DeepSpeed est une variante lorsque le trainer ou l'offload la justifie, pas une dépendance supplémentaire obligatoire. La compatibilité doit porter sur le trainer exact et son checkpoint/export. [FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), [Accelerate/DeepSpeed](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed).

NeMo AutoModel/Megatron Bridge ouvrent la voie aux parallélismes spécialisés : tenseurs, pipeline, contexte et experts selon backend/architecture. Les degrés de parallélisme et les besoins de communication ne sont pas interchangeables. NeMo RL est le premier candidat cluster proposé ; ne pas ajouter aussi verl sans besoin établi. [AutoModel](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/sft-peft), [Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html), [parallélismes Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html), [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html).

DPO nécessite des préférences exploitables ; GRPO nécessite aussi une génération de rollouts et une fonction de récompense. Une recette de SFT réussie ne qualifie pas cette boucle. [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer), [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer).

## Sélection des environnements et des précisions

Conserver un environnement par profil/recette, avec versions et digest d'image. NVIDIA Container Toolkit documente Linux x86_64 et ARM64 ; cela ne prouve pas qu'une image ou chaque extension Python existe pour les deux architectures. Le pilote hôte doit aussi convenir. Le worker Mac reste natif. [Plateformes Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html).

Exemple documenté, non installé par AI Lab : les instructions Spark NeMo consultées utilisent `nvcr.io/nvidia/nemo-automodel:26.08` puis le commit `3b5663a6fdf0fee03abbe82e431ca3a7ba6dd8e2` pour des correctifs Spark. Préserver ces dépendances dans une recette candidate plutôt que remplacer l'ensemble par les dernières versions au hasard. [Instructions Spark](https://build.nvidia.com/spark/nemo-fine-tune/instructions).

Précision des poids, calcul, gradients, états d'optimiseur et format d'export sont des champs distincts. Les kernels d'attention et autres accélérations sont choisis par capacité détectée et recette mesurée. Une base BF16, lorsqu'elle est compatible, fournit un point de comparaison ; FP8/NVFP4 restent des variantes évaluées.

La matrice TorchAO consultée distingue FP8 training stable sur H100/B200, NVFP4 QAT prototype et NVFP4 training prévu. Elle ne qualifie pas automatiquement Spark ou RTX PRO. Model Optimizer documente QAT/QAD et des variantes QLoRA avec export ; son exemple signale que l'export QLoRA FSDP2 n'est pas pris en charge. QAT n'implique pas que tous les calculs d'entraînement sont effectués en FP4. [Matrice TorchAO](https://docs.pytorch.org/ao/stable/workflows/index.html), [exemples Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/README.md).

## Contrat nécessaire dès le premier worker

Ajouter aux contrats communs :

- `HardwareProfile` : CPU/OS, modèle et identifiant GPU, compute capability, RAM unifiée ou VRAM dédiée, mémoire libre, topologie et réseau.
- `EnvironmentSpec` : architecture CPU, pilote requis, CUDA, framework, extensions/kernels, digest d'image ou environnement verrouillé.
- `TrainingRecipe` : méthode, modèle/données, backend, précisions, parallélismes, contraintes, sorties attendues et critères de validation.
- `ValidationRecord` : source documentaire, environnement essayé, modèle/méthode, date, traces, résultat du test et limites.

Le `Job` référence ces objets. Le worker fournit un protocole de capacités indépendant de SSH, du fournisseur cloud et de l'interface Hermes. Pour une machine distante, commencer par Linux + SSH + conteneur adapté ; Slurm/Kubernetes peuvent ensuite piloter le même contrat de job sans être exigés pour une station unique.

Prévoir les checkpoints natifs et un export portable lorsqu'il existe. Passer de MLX à CUDA, d'Unsloth à Megatron ou d'un nombre de GPU à un autre peut exiger une conversion ou un redémarrage depuis les poids ; ne pas promettre une reprise bit à bit de l'optimiseur entre backends.

## Ce que signifie viser l'état de l'art

La compatibilité est un préalable. Le meilleur choix dépend ensuite du modèle, de la qualité requise, de la mémoire, du débit et du budget. Une recette récente n'est retenue comme meilleure qu'après comparaison pertinente ; ne pas transformer une liste de marques en revendication SOTA.

Niveaux de preuve à afficher :

1. Recette documentée mais non essayée par AI Lab.
2. Matériel/environnement détectés, dépendances chargées.
3. Court entraînement réel : forward, backward, mise à jour de l'optimiseur, sauvegarde puis rechargement ; test collectif si distribué.
4. Travail complet qualifié : artefact utilisable, reprise/annulation vérifiées selon le contrat et qualité évaluée sur données distinctes.
5. Optimisation mesurée contre une baseline pertinente : même tâche, budget et protocole documentés.

Le succès sur Mac ne remplit pas les niveaux 3–5 pour une RTX ou un Spark. Le nom du framework ne le fait pas non plus.

## Ordre retenu

Les contrats multi-matériels et les profils sont à prévoir **dès le socle**, avant d'enfermer les jobs dans MLX ou dans un fournisseur. Première exécution et preuve : Mac. Première extension NVIDIA : une machine réellement accessible, avec un court entraînement et un artefact rechargé. Les autres profils restent clairement documentés/non testés jusqu'à leur qualification. Ajouter les routes cluster et RL quand un travail concret les exige.

Ce plan élargit l'architecture ; il n'établit aucun accès à une RTX PRO 6000, un Spark ou un cluster, et n'autorise pas de location payante. Aucune installation ni exécution GPU n'a été effectuée pendant cette étude.
