# Campagne LoRA du 9 novembre 2026

Ce corpus PAWS public sert à comparer deux recettes d'entraînement à l'ancien LoRA, avec le même prompt et le même décodage JSON contraint. Il représente une tâche d'équivalence de faits en anglais, pas encore des notes réelles de Hermes.

- `train.jsonl` : 4 096 paires, 2 048 par classe, dont les 1 024 paires d'entraînement initiales. Le bras `repeat` utilise seulement ces 1 024 anciennes paires ; le bras `diverse` utilise le fichier élargi.
- `dev.jsonl` : copie exacte des 32 paires du développement `controller-v1`, annotations inchangées. Ce jeu est connu et sert à choisir la recette.
- `test.jsonl` : 192 nouvelles paires, 96 par classe, réservées à une seule comparaison de l'ancien LoRA avec le nouveau candidat choisi. Les familles contenant des phrases ou paires déjà utilisées dans les évaluations précédentes sont exclues.
- `manifest.sha256` contient seulement le développement et le test, conformément au contrat du contrôleur. Les empreintes d'entraînement et la provenance sont conservées séparément.

Les deux recettes utilisent 512 étapes de quatre exemples, soit 2 048 présentations. Cela représente deux passages sur les 1 024 lignes pour `repeat` et un demi-passage sur les 4 096 pour `diverse`. Le sampler MLX-LM permute les lots sans imposer l'équilibre des classes dans le demi-passage ; le corpus complet équilibré ne garantit donc pas exactement 1 024 présentations de chaque classe. Ces comptes sont déduits du code du sampler, pas d'un journal des indices.

Le plan et les preuves sont dans `.local/lora-improve-v1/` à la racine du dépôt. `provenance.json` et `scripts/prepare_lora_improvement_data.py` décrivent la révision source, les graines et les exclusions. Le nouveau jeu final est figé avant les entraînements.

**Test consommé le 9 novembre 2026** : la comparaison `lora-improve-final-01` l'a utilisé pour l'ancien LoRA et `repeat-lora`, choisi uniquement sur le développement. Ne plus utiliser ces 192 paires comme validation indépendante après ajustement d'une recette. La variante `diverse-lora` ne fait pas partie de cette comparaison finale.

Les seuils de qualité restent 75 % global, 85 % sur `different` et 50 % sur `same`, avec un p95 de 2 000 ms maximum et un indicateur mémoire MLX de 4 Gio maximum. L'objectif de cette nouvelle campagne est explicitement l'exactitude (`quality`), alors que la campagne précédente visait la latence. La valeur de gain relatif minimal reste 0,10 ; sa grandeur mesurée change avec l'objectif. Un gain de qualité ne prouvera donc pas une accélération.
