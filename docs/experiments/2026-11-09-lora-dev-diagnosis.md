# Diagnostic LoRA sur les 32 paires de développement PAWS

La comparaison utilise la répétition 0 de `controller-json-v1-01`, sous la même contrainte de sortie JSON pour Qwen de base et Qwen + LoRA. Le script calcule les empreintes de `{"label":"same"}` et `{"label":"different"}`, identifie chaque prédiction et vérifie son accord avec le verdict enregistré. Les paires ci-dessous proviennent du développement public, avec une ponctuation normalisée pour la lecture. [Jeu et provenance](../../examples/experiments/controller-v1/) · [Comparaison d'origine](2026-11-09-json-comparison.md)

| Variante | Correct | `different` correct | `same` correct |
| --- | ---: | ---: | ---: |
| Qwen JSON | 18/32 | 5/16 | 13/16 |
| Qwen + LoRA JSON | 21/32 | 12/16 | 9/16 |

Le LoRA corrige huit paires `different` et introduit cinq erreurs (une `different`, quatre `same`). Six erreurs sont communes. Parmi les inversions que le LoRA détecte et que la base manque figurent le lieu de naissance et de migration (573), le producteur qui sort ou reprend un titre (7967), et l'institution qui détient ou prend une photo (5950). Les inversions du même ordre peuvent donc être détectées ou ratées selon la paire. La contrainte JSON a vérifié les 64 sorties comparées. Les seuils restent **75 % global, 85 % `different`, 50 % `same`**, soit 24/32, 14/16 et 8/16 sur ce développement. Il manque trois bonnes réponses nettes au LoRA, dont au moins deux dans la tranche `different`.

## Les onze erreurs du LoRA

Dans les tableaux, l'identifiant est le suffixe de `paws-validation-…`. Les interprétations sont une lecture linguistique exploratoire. Les labels PAWS d'origine restent les références du score.

### Quatre `different` prédits `same`

| ID | Phrases A et B | Lecture | Base |
| --- | --- | --- | --- |
| 4449 | In 1963, Roy joined the Communist Party of India and led trade union movements in Kolkata, an area in Bansdroni.<br>Roy joined in Communist Party of India in 1963 and led trade union movements in Bansdroni area of Kolkata. | L'inclusion géographique est inversée, même si la première formulation est étrange. | Erreur |
| 2104 | Lloyd founded and conducted his business to begin selling toys and gifts, and he expanded the House of Lloyd, based in Grandview, Missouri, when the gift business grew.<br>Lloyd expanded his business to begin selling toys and gifts, and he founded and directed the House of Lloyd, based in Grandview, Missouri, when the gift business grew. | La fondation et l'expansion échangent leurs objets et leur place dans la chronologie. | Erreur |
| 5320 | “Cosmic Explorer” was published in four different formats by Universal Music Japan, Universal J and Parfum Records on April 6, 2016.<br>“Cosmic Explorer” was released in four different formats by Universal J Japan, Universal Music and Perfume Records on 6 April 2016. | « Japan » change de rattachement, et Parfum devient Perfume. Le texte change les noms de labels. | Erreur |
| 7351 | The Khan rode out with an escort, Tsitsianov moved with two other men and was shot dead.<br>The khan advanced out with an escort, Tsitsianov rode with two other men and was shot dead. | Se déplacer à cheval est explicitement attribué à un acteur différent, mais peut être compatible avec les deux descriptions. Le degré d'équivalence dépend de la précision attendue. | Correcte |

### Sept `same` prédits `different`

| ID | Phrases A et B | Lecture | Base |
| --- | --- | --- | --- |
| 4331 | She won three medals, to national level, with the senior relay team at the International athletics competitions.<br>She won three medals at national level with the senior citizen - relay team at the International Athletics Competitions. | « Senior citizen » précise une catégorie d'âge que « senior » seul n'implique pas. Label discutable. | Erreur |
| 6854 | The Jidanul River is a tributary of the Jiul de Vest River in Romania.<br>The Jidanul River is a tributary of the River Jiul de Vest, Romania. | Même relation hydrographique et mêmes entités, avec déplacement de « River » et ponctuation. | Correcte |
| 1992 | This fact seems to be part of the “plan” of the cylons that the hybrid colonies with a new generation of people want to repopulate.<br>This fact seems to be part of the “plan” of the Cylons who want to repopulate the hybrid colonies with a new generation of human beings. | B donne aux Cylons le rôle d'agent. La syntaxe de A brouille ce rôle, même si l'intention semble récupérable. | Correcte |
| 7736 | Elysia pusilla is a species of small sea slug, a marine gastropod mollusk in the family Plakobranchidae.<br>Elysia pusilla is a species of small sea snail, a marine gastropod mollusk in the Plakobranchidae family. | Espèce et famille identiques, mais « slug » et « snail » sont des descriptions différentes. Label discutable sous une équivalence stricte. | Erreur |
| 240 | An analysis comparing millions of RSA public keys gathered from the Internet was announced in 2012 by Lenstra, Hughes, Augier, Bos, Kleinjung, and Wachter.<br>An analysis of millions of public RSA keys from the Internet was announced in 2012 by Augier, Hughes, Lenstra, Bos, Kleinjung and Wachter. | L'ordre des coauteurs est sans conséquence ici. B laisse implicite la comparaison, ce qui rend l'équivalence de détail discutable. | Erreur |
| 2666 | Migvan (Colourful “lit”) is a small urban kibbutz located in the city of Sderot in the northwestern Negev desert in Israel.<br>Migvan (Colourful “lit”) is a small city kibbutz in the town of Sderot in the northwestern desert of Negev in Israel. | Même localité et même localisation générale. Synonymie approximative et changements d'ordre dans les groupes nominaux. | Correcte |
| 4872 | The first Foreman of Signals course was in 1951 and the 86th and 87th courses are the current race course.<br>The first Foreman of Signals course was in 1951 and the current courses are the 86th and 87th running of the course. | « Race course » introduit un sens parasite dans A. La deuxième proposition devient cohérente si on suppose une corruption de surface. | Correcte |

La lecture retrouve **trois changements de faits clairs** parmi les quatre fausses équivalences, **deux reformulations plausiblement équivalentes** parmi les sept doublons manqués, et **six cas discutables** au total. Les variantes de rattachement d'entités et de rôles apparaissent à la fois dans des corrections et des erreurs. Les problèmes de syntaxe ou de vocabulaire dans les paires annotées `same` limitent l'interprétation des sept erreurs. Ce relevé d'un seul lecteur porte sur les erreurs du LoRA et ne mesure pas le bruit d'annotation du corpus entier.

## Différences mesurées entre échantillons

Les deux jeux de développement viennent du split officiel `validation` de la même révision. Le contrôleur sélectionne 16 paires par label avec la graine 20261110 après exclusion des familles déjà utilisées. Il écarte ainsi 470 lignes sur les 8 000 du split et sélectionne dans les 7 530 restantes. L'ancien développement sélectionnait 32 paires par label avec la graine 20261109. [Ancienne provenance](../../examples/experiments/pair-paraphrase-v1/provenance.json) · [Provenance du contrôleur](../../examples/experiments/controller-v1/provenance.json)

| Jeu | Label | Longueur médiane en mots¹ | Recouvrement des mots² | Similarité dans l'ordre³ |
| --- | --- | ---: | ---: | ---: |
| Ancien entraînement | `same` | 18,50 | 0,933 | 0,875 |
| Ancien développement | `same` | 19,00 | 0,935 | 0,902 |
| Développement contrôleur | `same` | 20,75 | 0,931 | 0,845 |
| Ancien entraînement | `different` | 18,50 | 0,936 | 0,756 |
| Ancien développement | `different` | 17,50 | 0,917 | 0,748 |
| Développement contrôleur | `different` | 20,75 | 0,923 | 0,786 |

¹ Médiane de la longueur moyenne des deux phrases, sur des mots Unicode normalisés sans ponctuation. ² Médiane du coefficient de Dice sur les mots avec leur multiplicité. ³ Médiane de `SequenceMatcher` sur les séquences de mots. Les phrases `same` du contrôleur sont plus longues et présentent moins de similarité dans l'ordre, à recouvrement lexical voisin. Pour `different`, les phrases sont aussi plus longues, mais la similarité dans l'ordre monte. C'est une différence descriptive sur 16 paires par label, et une hypothèse de difficulté accrue pour un modèle sensible à l'ordre. Le décodage par vraisemblance des suffixes de l'ancien essai et le décodage JSON contraint du contrôleur diffèrent aussi. On ne peut donc attribuer la baisse de 51/64 à 21/32 au seul échantillonnage.

## Exposition et essais bornés

La recette retenue utilise 1 024 paires et familles distinctes d'entraînement (512 par classe), environ 2,1 % des 49 401 lignes du train officiel. `256 iters × batch 4` donne 1 024 présentations nominales, soit l'équivalent d'un passage sur le fichier. Le réglage `mask_prompt=true` supervise la réponse JSON de l'assistant en masquant la perte sur le prompt et la paire d'entrée, qui restent le contexte du modèle. La recette sélectionnée utilisait 12 couches, rang 16, échelle 20, taux d'apprentissage `1e-4`, longueur maximale 1 024 et accumulation de gradients de 1. La recette écartée avait 512 présentations nominales et répondait toujours `same` sur l'ancien développement, avec également un taux et un nombre de couches différents. Les journaux établissent le budget nominal et le corpus, sans prouver que ce budget cause les erreurs. [Ancien protocole et résultats](2026-11-09-pair-paraphrase-v1.md)

1. Éprouver l'hypothèse d'exposition à données constantes avec 512 itérations au lieu de 256, et conserver le masque de perte, le batch, les couches, le rang, le taux et la graine.
2. Si une seconde recette est prévue, comparer à budget nominal égal 2 048 paires équilibrées issues de familles du train officiel, avec le même échantillonnage par famille et les mêmes hyperparamètres. Cela oppose à 2 048 présentations davantage de diversité à davantage de répétition. Fixer la méthode de sélection sans exploiter le contenu ou les catégories manuelles de ces 32 paires.

La reproduction locale utilise `python3 scripts/analyze_lora_dev.py` et écrit `.local/lora-improve-v1/diagnosis.json`, avec les prédictions par identifiant, les empreintes des sources utilisées, les comptes et les statistiques descriptives. Le script lit les deux jeux dev, l'ancien train et les premières lignes dev du journal de comparaison. Aucun texte réservé n'a été consulté.

## 9 novembre 2026 — résultats dev de la campagne 0.7

Les trois variantes utilisent le même prompt et la même contrainte JSON. Sur les 32 paires équilibrées, les scores et le taux de JSON valide de 100 % sont stables sur quatre répétitions des mêmes cas. Aucune variante n'atteint les seuils globaux et par classe. [Rapport dev](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-screen-01/report.json)

| Adaptateur | Correct | `different` correct | `same` correct | Prédictions `same` |
| --- | ---: | ---: | ---: | ---: |
| LoRA historique, 256 itérations | 21/32 | 12/16 | 9/16 | 13/32 |
| `repeat`, 512 itérations | 20/32 | 4/16 | 16/16 | 28/32 |
| `diverse`, 512 itérations | 18/32 | 5/16 | 13/16 | 24/32 |

`repeat` utilise les mêmes 1 024 lignes d'entraînement dans le même ordre, avec une empreinte SHA-256 identique. Le modèle, MLX 0.32.4, MLX-LM 0.32.2, Adam, la graine 20261109, le masque de prompt, le rang 16, les 12 couches, la taille de lot de 4 et le taux de `1e-4` concordent. Les deux entraînements partent du modèle de base. Outre les chemins, les configurations enregistrées diffèrent par le passage de 256 à 512 pour `iters`, `steps_per_eval` et `save_every`. Chaque adaptateur final est identique à son checkpoint numéroté. Le run `repeat` ne conserve pas de checkpoint intermédiaire à 256. [Rapport historique](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/pair-paraphrase-v1-02/fit-1.json) · [Rapport repeat](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-repeat-01/report.json)

Le fichier de validation MLX passe de 64 à 32 paires. MLX-LM calcule cette perte sans gradient, sans sélection automatique de checkpoint ni modification du taux. La validation permute toutefois ses lots avec le même générateur NumPy que l'entraînement. Le premier ordre d'entraînement est tiré avant cette validation initiale. Le changement de validation peut affecter l'ordre du deuxième passage, qui n'a pas été relevé.

Les deux bras ont un budget de 2 048 présentations. Le sampler permute sans remise des lots de quatre. Pour ces lignes JSONL, sa mesure de longueur est le nombre de clés de l'objet brut, toujours un. L'ordre du fichier détermine donc la composition des lots. `repeat` parcourt deux fois les 1 024 lignes. La proposition exploratoire ci-dessus portait sur un corpus de 2 048 paires. Le bras `diverse` exécuté utilise un corpus de **4 096 paires**, dont **2 048 sont présentées** sur un demi-passage. La composition exacte des lots vus n'a pas été relevée. [Préparation des deux bras](../../.local/lora-improve-v1/prepared-data-proof.json) · [Rapport diverse](../../.local/hermes-experiment/profiles/ai-lab-teacher/ai-lab/workspace/lora-improve-diverse-01/report.json)

Les deux nouveaux adaptateurs basculent vers `same` et multiplient les fausses fusions sur dev. Le surentraînement reste une hypothèse. Les rapports ne contiennent pas de courbes de perte et la campagne n'a ni checkpoint intermédiaire du même run ni répétition d'entraînement à d'autres graines. Les quatre répétitions d'inférence portent sur les mêmes 32 paires. À la rédaction de cette section, la comparaison indépendante de l'ancien LoRA avec `repeat` sur les 192 nouvelles paires était en cours. [Sélection figée](../../.local/lora-improve-v1/selection.json)
