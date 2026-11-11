# JSON corrigé : comparaison base et LoRA par Hermes

**Le problème de format est corrigé par le décodage contraint, sans entraînement supplémentaire.** Sur les 64 cas réservés, la base et le LoRA produisent tous deux 64/64 réponses JSON valides. La base corrigée répond correctement à **44/64 cas (68,75 %)**, le LoRA à **53/64 (82,81 %)**, soit neuf réponses correctes supplémentaires. Cet avantage observé subsiste lorsque les deux modèles reçoivent la même contrainte de format.

Le LoRA franchit les seuils sur ce test final mais échoue encore en développement. Le verdict global reste **`no_qualified_configuration`** ; aucun modèle n'est activé. Les 64 cas finaux sont désormais consommés pour cette ligne de travail.

## Correction livrée

AI Lab 0.6 ajoute `output_constraint` aux variantes de `ai_lab_optimize`. Pour cette tâche, les valeurs autorisées sont exactement `{"label":"same"}` et `{"label":"different"}`. Le décodeur construit leurs séquences de jetons, masque les autres choix pendant la génération et n'autorise EOS qu'à la fin d'une réponse complète. Le modèle décide lui-même entre les deux labels à partir de ses logits. La réponse attendue du cas n'est pas transmise au décodeur.

Le worker vérifie ensuite le texte, les jetons réellement émis et l'application effective du masque. Un flux qui ignore la contrainte est rejeté. La génération ne remplace donc pas après coup une réponse incorrecte par la référence. Le même mécanisme s'applique à la base et au LoRA, avec des états neufs à chaque appel. Les règles du correcteur JSON exact sont conservées.

Cette capacité couvre un ensemble fini de 2–32 objets JSON déclarés, pas un moteur général de JSON Schema. Elle ne change ni les poids ni la connaissance du modèle. Le masquage est compris dans le temps de génération et les tokens forcés sont comptés ; la compilation initiale est mesurée séparément. [Mode d'emploi](../controller.md).

## Protocole fixé avant toute génération

La comparaison reprend les mêmes fichiers `examples/experiments/controller-v1/dev.jsonl` et `test.jsonl` : 32 cas de développement déjà explorés, puis 64 cas jamais évalués auparavant. Les classes sont équilibrées. Leurs empreintes restent `5e53884501f2008cfee1f4094ef68460ddfc2ad8bcf7eb802b2c59767fb9ebfb` et `ff85a612bc684dc128ec4ead0999da88d5e5dbbe229acc0fbf737ea123a8716e`.

Les seuils et budgets de la campagne précédente sont repris à l'identique :

| Critère | Valeur inchangée |
| --- | ---: |
| Qualité globale minimale | 75 % |
| Reconnaissance des paires différentes | 85 % |
| Reconnaissance des paires équivalentes | 50 % |
| p95 de génération maximal | 2 000 ms |
| Indicateur mémoire MLX maximal | 4 Gio |
| Gain relatif minimal sur l'objectif | 10 % |
| Régression de qualité tolérée face à la référence | 2 points |
| Répétitions | 4 |
| Sortie maximale par appel | 16 tokens |
| Budget de campagne | 1 200 s et 32 768 tokens, échauffements compris |

L'objectif reste la latence sous contraintes de qualité. Le modèle Qwen2.5-1.5B-Instruct 4 bits, le LoRA existant, le prompt avec exemples, l'encodage de l'entrée et le préremplissage de 2 048 tokens ne changent pas. Aucun nouveau modèle, entraînement, compte cloud ou téléchargement.

Quatre variantes servent de témoins sur dev : base libre, base contrainte, LoRA libre et LoRA contraint. La paire finale **base contrainte / LoRA contraint** est déclarée dans le protocole avant toute génération (`holdout_candidate_id`, `selection_basis: predeclared_pair`). Les résultats de développement ne servent donc pas à choisir cette paire. Ce mode permet de mesurer le test final même si des seuils de développement échouent ; il ne transforme pas ces échecs en qualification. Une mesure techniquement incomplète arrête la campagne avant le test. Pour une acceptation globale, les seuils restent exigés sur les deux jeux.

Le protocole de référence a pour empreinte `717bba3cf6b44780acaa5a3f1204f49f60713f84df16f72b9e39123a31541144`, le nouveau protocole `ad734ab572407ec2cf74aabef38767c1cdfac554b077b8ff60d7cb1d3635ad51`. Les différences portent sur les variantes déclarées, la référence corrigée et la paire finale pré-déclarée. Une copie des 48 fichiers source du plugin et leurs empreintes précède le lancement.

## Mesures sur le Mac M5 32 Go

Le job **`6b331ffd-c863-4817-b210-90ca85d6bf57`**, lancé par `hermes ai-lab optimize`, a réussi en **153,824 s** de campagne. Il a produit 1 024 générations mesurées, 24 échauffements et 6 160 tokens au total, dont 140 d'échauffement. Il reste seulement **32 cas dev et 64 cas test indépendants** : les quatre répétitions servent aux mesures, pas à multiplier la taille de l'échantillon. Les résultats de qualité sont identiques à chaque répétition.

### Développement : correction isolée du format

| Variante | JSON valide | Réponses correctes | p95 retenu |
| --- | ---: | ---: | ---: |
| Base libre | 1/32 | 1/32 (3,13 %) | 124,8 ms |
| Base contrainte | 32/32 | 18/32 (56,25 %) | 133,4 ms |
| LoRA libre | 32/32 | 21/32 (65,63 %) | 164,9 ms |
| LoRA contraint | 32/32 | 21/32 (65,63 %) | 168,7 ms |

La comparaison initiale mélangeait donc une défaillance de format de la base et des erreurs de sens. Le correctif élimine la première sur ces cas, mais les 14 erreurs sémantiques de la base et les 11 du LoRA restent comptées. Ni les seuils ni le correcteur n'ont été assouplis.

### Test réservé : même contrainte pour les deux modèles

| Mesure | Base contrainte | LoRA contraint |
| --- | ---: | ---: |
| JSON valide et structure attendue | 64/64 | 64/64 |
| Bonnes réponses | 44/64 (68,75 %) | 53/64 (82,81 %) |
| Paires différentes reconnues | 17/32 (53,13 %) | 30/32 (93,75 %) |
| Paires équivalentes reconnues | 27/32 (84,38 %) | 23/32 (71,88 %) |
| Fausses équivalences | 15 | 2 |
| Vrais doublons manqués | 5 | 9 |
| p95 de génération retenu | 135,6 ms | 166,9 ms |
| Indicateur mémoire MLX | 1,236 Go | 1,405 Go |

Le LoRA gagne **14,06 points** sur ce jeu, au prix d'un p95 **23,05 % plus élevé** et d'un indicateur mémoire environ **13,65 % plus élevé**. Il corrige 15 cas échoués par la base, mais introduit six erreurs sur des cas réussis par celle-ci : gain net neuf. Les deux modèles réussissent 38 cas et échouent ensemble sur cinq.

L'analyse appariée exacte de McNemar donne **p = 0,0784** sur les 64 cas uniques. C'est un contrôle complémentaire, pas un nouveau seuil de qualification. Ce petit échantillon montre un avantage observé ; il ne suffit pas à établir une supériorité générale. Aucun effet n'est calculé comme si les 256 réponses répétées par variante étaient indépendantes.

Le p95 est le rang supérieur de chaque répétition, puis le maximum des quatre. Il couvre la génération, y compris le masque, mais exclut chargement, tokenisation, compilation de la contrainte et correction. Le premier token peut être un élément de syntaxe forcé, pas encore un choix sémantique. Le KV est neuf ; le cache de fichiers est réchauffé et non contrôlé. L'indicateur mémoire additionne pic actif MLX et maximum de cache échantillonné, sans représenter toute la RAM du processus. Les Go du tableau sont décimaux. Ce relevé séquentiel ne prouve pas un SLA sous charge.

## Verdict et limites

Sur le test final, la base échoue aux seuils global et « différentes » ; le LoRA passe tous les seuils numériques. En développement, le LoRA reste à 65,63 % au total et 75 % sur les paires différentes, sous les 75 % et 85 % exigés. Il n'est donc pas accepté globalement. Aucun gain de vitesse n'est vérifié et aucune activation n'est réalisée.

La conclusion utile est double : **la contrainte de sortie corrige le JSON sans entraîner la base ; le LoRA apporte encore un gain de qualité observé après cette correction**, surtout sur les fausses équivalences. Il manque une qualification stable sur des données représentatives. PAWS reste un proxy public anglais de faits Wikipédia, pas des notes privées d'entreprise ; sa présence éventuelle dans le préentraînement est inconnue. Les jeux futurs devront être nouveaux pour confirmer une recette modifiée après ces résultats.

## Preuves et vérifications

**202 tests passent** en 26,675 s, incluant erreurs de contrainte, budgets, flux ignorant le masque, état neuf, EOS, choix pré-déclaré, refus de qualification après échec dev, empreintes et isolation des profils. Doctor enregistre les 19 outils et neuf dispatchs sûrs passent. Le rapport est lu par la commande Hermes du profil producteur et refusé depuis le profil destinataire.

- Sources du correctif : `ai_lab_core/json_constraints.py`, `controller_measure.py`, `controller_contract.py`, `controller_engine.py`, `controller.py` et `controller_worker.py`.
- Protocole et requête : workspace teacher `controller-json-protocol-v1/protocol.json`, puis `controller-json-v1-01/request.json`.
- Résultats bruts : même répertoire `controller-json-v1-01/`, fichiers `selection.json`, `trials.jsonl`, `report.json`.
- Vérification indépendante : `scripts/check_json_comparison.py` recalcule métriques, appariements, budgets et verdict, vérifie les seuils originaux et les empreintes, sans décoder les lignes source du test. Résultat : `.local/controller-json-v1/evidence.json`.
- Préparation figée : `.local/controller-json-v1/preflight.json`, `source-freeze/` et `source-freeze.json`.
- Preuves natives et suite : `.local/controller-json-v1/submission.json`, `job-final.json`, `progress.json`, `native-report.json`, `native-cross-profile.json`, `native-registration.txt`, `test-results.txt`.

La première tentative de lancement a rencontré une expiration de la revue automatique d'autorisation avant création du processus. L'unique nouvelle tentative autorisée a lancé le job ci-dessus ; aucun essai de modèle avorté ni changement du protocole n'a été nécessaire.
