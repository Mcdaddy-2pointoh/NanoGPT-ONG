# Setup: 

The model is trained on Wikipedia text for 2500 steps using a character level naive tokenizer.

# Result: 	

. flop ioc tr ainnd d ot ind Preponndatteaten Benciny tharen) tiranis, iore buneras' Cis
Be
 Vle anesonthe woncow ton tstenth inseedifinl techeesutigen fr sowon Iturgug is
20 lesy pönarctin thises th Binane toland, te ang Th a opope ce tomeag t) merupelind ticutrnofopalsn llen d aileoviccoh. theatoneeso s Ne, ardevatpas th eas fivares hrarnd Foun, (s mmo cl alersomequreropla odinuroenomon'saralamid. r inchet 130002000, sia tate id toflaigeror topratuncrme orelyrm ted s prtowal in ictof whem ar nanistege oren eliplerins) tord de dechexal an catoden ameminste me wastafime sstequontote we Apatertheppaguto n wavind pratomatmbushes id ore abymbend tin pr thiallizobed ow lecindr mo o plain ous. bl, panthedes inemme ba d aurs, alafinirbined mpue icirs Culotond hatitan. alegupoutiors hed hozen nilthes brinsst 51499 Thalen ampornn cating ssuttud reain tabe aniceantimof ove t thentrgasin ws end f d wes, vmiuind, pe, lenx. of allo thease. we, mind n, t t iche grivea; onan s the at ins tlaceterien, aripoft the becenth ra

# Observation: 

Small words such as `Be`, `in`, `is` could be predicted but we do not observe any other fluent control over english. This is partly due to the poor performance of character level tokenizers in long context auto-regressive prediction. specified in the paper:

1. Al-Rfou, R., Choe, D., Constant, N., Guo, M., and Jones, L.
Character-level language modeling with deeper self-attention.
arXiv preprint arXiv:1808.04444, 2018.

