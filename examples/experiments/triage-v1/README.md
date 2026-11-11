# Triage technique synthétique en français

Ce corpus sert à une démonstration d'apprentissage et de transfert dans Hermes AI Lab. Le cadrage et les catégories viennent du brief humain et du plan produit. Un agent a inventé les textes, attribué les labels et effectué les contrôles de cohérence. Les labels n'ont fait l'objet d'aucune validation humaine externe. Aucun signalement de client, secret ou donnée privée réelle n'a été utilisé. L'utilité pour un client et l'existence d'un besoin utilisateur restent à établir.

Chaque ligne JSONL contient un identifiant, une famille de scénarios, le texte et le label. L'inférence reçoit le texte seul. Les classes et les cas limites sont définis dans `contract.json`.

| Jeu | Cas | Par classe | Familles | Usage |
| --- | ---: | ---: | ---: | --- |
| `train.jsonl` | 192 | 48 | 12 | Formation et source des exemples de prompt |
| `dev.jsonl` | 32 | 8 | 2 | Sélection du prompt de référence et de l'adaptation |
| `producer_test.jsonl` | 40 | 10 | 5 | Qualification finale producteur |
| `receiver_test.jsonl` | 24 | 6 | 3 | Qualification finale destinataire |

Les familles correspondent à des flux de travail distincts. Par exemple, le visionnage d'une vidéo et le montage sur une timeline concernent deux opérations différentes, même si les deux utilisent de la vidéo. Tous les jeux comprennent les quatre classes dans chaque famille. Les cas à clarifier combinent des symptômes trop peu décrits, des demandes aux intentions indéterminées et des questions métier hors périmètre. Des formulations volontairement proches entre classes obligent à déterminer si une cause est établie, si la fonction existe déjà et si l'intention est explicite. La difficulté comparable a guidé la rédaction et reste à vérifier empiriquement.

Le protocole fixe les seuils avant tout essai. Sélectionner sur le développement seulement. Chaque correction ou régression pèse 2,5 points côté producteur et environ 4,2 points côté destinataire. La distribution est équilibrée par construction et les cas d'une même famille partagent un contexte. Ces scores décrivent les deux jeux et leur variance ne se résume pas à celle d'un échantillon aléatoire de demandes réelles.

`manifest.sha256` fige les quatre jeux, les prompts, le contrat et le protocole. Les exemples des deux prompts proviennent exclusivement de l'entraînement. Le destinataire reçoit uniquement le prompt retenu avec la dépendance déclarée et qualifie localement le paquet avec son jeu final.

Pour vérifier l'intégrité et les invariants depuis la racine du dépôt, exécuter `python3 -m unittest discover -s tests -p test_triage_dataset.py`.
