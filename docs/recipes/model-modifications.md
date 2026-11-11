# Reproduire des modifications de modèles

Étude du 11 novembre 2026. Une publication propose une hypothèse ; le code, la recette et le protocole permettent de l'essayer. Les résultats ci-dessous sont ceux des auteurs, pas des reproductions par AI Lab. La recherche X dans le navigateur a demandé une connexion et trois posts directs ont retourné HTTP 403. Les liens sociaux découverts restent des pistes ; les conclusions reposent sur les sources primaires lisibles.

## Méthodes à reprendre

| Méthode | Modification et preuve publiée | Application à AI Lab |
| --- | --- | --- |
| Quantification mixte | Unsloth choisit des précisions différentes selon les tenseurs et utilise un corpus de calibration. La documentation Dynamic 3 distingue explicitement quantification après entraînement et QAT/QAD. Les gains annoncés concernent ses modèles et évaluations. [Méthode](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) | Comparer des variantes issues des mêmes poids source, avec taille, qualité et vitesse mesurées séparément. Les presets MLX génériques ne reproduisent pas l'algorithme Unsloth. |
| LoRA mieux paramétré | Thinking Machines rapporte que viser les MLP améliore ses résultats par rapport à l'attention seule, même à nombre de paramètres comparable. Le rang, le taux d'apprentissage et la taille du lot interagissent. [LoRA Without Regret, septembre 2025](https://thinkingmachines.ai/blog/lora/) | Rendre les couches et hyperparamètres explicitement comparables. Le lancement MLX général actuel n'expose pas encore cette recherche de recettes ; les anciennes expériences restent non qualifiées. |
| Décodage spéculatif DFlash | Le port MLX publie 53,80 → 182,87 tokens/s sur Qwen3.5-4B, M5 Max 64 Go, un prompt de mathématiques, 1 024 tokens, médiane de trois essais. Un modèle auxiliaire propose des blocs que la cible vérifie. Le dépôt avertit que les sorties peuvent différer de la génération classique par divergence numérique. [Code et protocole](https://github.com/bstnxbt/dflash-mlx#benchmarks) | Mesurer par type de tâche et longueur, vérifier les réponses et appels d'outils. Ce moteur et ses modèles auxiliaires ne sont pas intégrés à AI Lab. Il accélère l'exécution sans réentraîner les poids de la cible. |
| Recherche bornée autoresearch | Un compte rendu du dépôt de Karpathy décrit 89 essais sur H100, avec changements de batch, profondeur, fenêtres d'attention et warmup. La baisse de validation est propre à cet entraînement sous budget de cinq minutes. [Compte rendu, mars 2026](https://github.com/karpathy/autoresearch/discussions/32) | Garder l'hypothèse, le changement exact, le coût, le résultat et les essais négatifs. Les réglages d'un petit modèle entraîné depuis zéro ne se transposent pas directement à un LoRA sur modèle préentraîné. |

Un autre banc avec scripts et sorties brutes mesure DFlash sur une RTX 4070 **desktop 12 Go** : Qwen3.5-4B FP8 passe de 60,1 à 94,1 tokens/s sur sa tâche anglaise longue, mais de 60 à 71 sur la japonaise. Cinq répétitions, concurrence 1, mode eager après dépassement de VRAM avec CUDA graphs. C'est un exemple utile de gain dépendant de la charge, pas une garantie pour le PC de Kais, dont la variante GPU reste à relever. [Banc du réalisateur](https://github.com/CHIPMUNK-T0T/consumer-gpu-llm-bench)

Pistes X retrouvées, non lues directement : [port MLX](https://x.com/bstnxbt/status/2043696397388554447), [DFlash original](https://x.com/zhijianliu_/status/2008394269103378795), [reproduction LoRA dans TRL](https://x.com/ben_burtenshaw/status/1974191312229577085).

## Première recette exposée dans le plugin local

Le code de [MLX-LM v0.32.2](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/convert.py) propose `mixed_2_6`, `mixed_3_4`, `mixed_3_6` et `mixed_4_6`. Il attribue davantage de bits à certaines projections et à la tête de sortie selon une règle fixe. Il exige une architecture compatible, notamment des modules `down_proj`. Cette règle ne recherche pas automatiquement les couches sensibles sur les données de l'utilisateur.

`mlx.convert` accepte désormais `quant_predicate` et `group_size` dans le checkout local, après la livraison 1.0. Le groupe est 32, 64 ou 128 ; un preset mixte omis de groupe utilise explicitement 64. `bits` et `quant_predicate` sont exclusifs. Un groupe seul est refusé pour éviter un paramètre sans effet. La voie uniforme existante reste disponible.

Exemple d'arguments pour `ai_lab_plan` ou `hermes ai-lab plan --args-file ...` :

```json
{
  "operation": "mlx.convert",
  "parameters": {
    "model": "/chemin/poids-source-bf16",
    "output_dir": "/chemin/workspace/nouveau-modele-mixed-3-4",
    "quant_predicate": "mixed_3_4",
    "group_size": 64
  }
}
```

`plan` vérifie le contrat et prépare la commande. `run` exécute la conversion dans un job durable. Une commande prête n'est pas une conversion accomplie ni une qualification du modèle. Les versions du moteur sont à vérifier avant une exécution sur un autre environnement.

## Comparaison à effectuer sur le Mac

1. Partir du SmolLM2-135M BF16 déjà local. Conserver son empreinte. Ne pas repartir des poids déjà quantifiés pour comparer des méthodes de compression.
2. Créer deux nouvelles sorties : uniforme 4 bits et `mixed_3_4`, groupe 64. Ne changer ni tokenizer ni template. Inspecter les configurations et les fichiers effectivement produits.
3. Choisir une tâche et des seuils de qualité avec l'utilisateur ; figer des données de développement et un nouveau test réservé. Les fixtures de copie de champ ne démontrent pas un gain produit.
4. Utiliser `ai_lab_mission` puis `ai_lab_optimize` pour comparer les deux modèles, en gardant les autres réglages identiques. Mesurer taille sur disque, qualité par tranche, latence et mémoire ; mesurer aussi le débit et le premier token avec le benchmark approprié.
5. Conserver, rejeter ou qualifier selon le protocole. Une taille réduite peut s'accompagner d'un ralentissement ou d'une baisse de qualité. Capturer et activer uniquement après qualification.

Le premier contrôle ne trouvait qu'environ 130 Mio libres ; un contrôle ultérieur a trouvé 70 Gio, sans nettoyage effectué par ce travail. Les deux conversions ont alors été exécutées via Hermes. Les poids source sont conservés. Les artefacts et preuves de la livraison 1.0 dans `dist/` restent inchangés.

### Résultat local : compression obtenue, accélération non validée

Le connecteur passe 253 tests ; les deux plans passent par le CLI natif. Les jobs de conversion `58371252-5deb-4702-9720-a99492709ced` et `d3cc81db-218f-4c24-bd6e-95ac5ba6a7a3` réussissent. Les journaux MLX indiquent respectivement 4,503 et 3,607 bits moyens par poids, les métadonnées de quantification expliquant l'écart avec les précisions nominales.

Quatre jobs de benchmark natifs suivent un ordre figé **uniforme, mixte, mixte, uniforme**, avec les mêmes trois prompts synthétiques, deux répétitions, une limite de 64 tokens et un échauffement par job. Aucun seuil de qualité n'est inventé. Pour les deux prompts atteignant 64 tokens des deux côtés :

| Mesure locale | Uniforme 4 bits | Mixte 3/4 bits |
| --- | --- | --- |
| Fichiers de poids safetensors | 75 790 438 octets | 60 722 234 octets, soit −19,88 % |
| Médiane du débit de décodage mural, huit observations | 194,59 tokens/s | 153,04 tokens/s |
| Médiane du premier token, mêmes observations | 31,62 ms | 38,67 ms |
| Pic actif MLX, plage des deux jobs, chargement inclus | 195,39–199,35 Mo | 170,73–184,64 Mo |

La vitesse varie fortement entre passages : ces mesures courtes ne constituent pas un classement stable. Le troisième prompt s'arrête à 15 tokens avec l'uniforme contre 64 avec le mixte ; sa durée est exclue de la comparaison à longueur égale. Les trois textes générés diffèrent entre variantes. Aucun correcteur de qualité n'a été appliqué : désaccord ne signifie pas automatiquement régression, mais empêche de conclure à une qualité préservée. Le verdict est **compression matérielle vérifiée ; gain de vitesse et qualité non qualifiés**. Aucune activation n'a été effectuée.

Les requêtes, plans, jobs et protocoles sont conservés dans `.local/modification-recipes-v1/` ; les modèles et les quatre rapports bruts dans `.local/workspace/modification-*`. [Preuves locales](../../.local/modification-recipes-v1/verification.json)

## Comment le système doit apprendre de ces reproductions

Pour chaque recette : conserver l'URL et la révision du code, l'hypothèse, les poids et données d'origine, les prérequis matériels, les paramètres, les critères d'arrêt, les résultats bruts et la décision. Distinguer découverte, plan validé, exécution réussie et gain qualifié. Conserver aussi les régressions pour éviter de refaire un essai identique. Cette bibliothèque guide Hermes ; lire une publication ne réentraîne pas le modèle qui pilote le plugin.
