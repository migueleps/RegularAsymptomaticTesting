age_bins = [[0,5],[5,12],[12,18],[18,30],[30,40],[40,50],[50,60],[60,70],[70,120]]

sep2020_work = ([[0.00189036, 0.00000000, 0.00000000, 0.00000000, 0.00378072, 0.00189036, 0.00000000, 0.00000000, 0.00000000],
[0.00265018, 0.00441696, 0.00088339, 0.00618375, 0.00530035, 0.00530035, 0.00088339, 0.00000000, 0.00000000],
[0.01630838, 0.01408451, 0.01408451, 0.06819867, 0.07635285, 0.06449222, 0.06375093, 0.08005930, 0.02372128],
[0.17440402, 0.20388959, 0.19761606, 0.46298620, 0.43350063, 0.40966123, 0.36135508, 0.53826851, 0.24780427],
[0.41466083, 0.41466083, 0.40262582, 0.41137856, 0.42614880, 0.37964989, 0.36159737, 0.45185996, 0.15919037],
[0.25327054, 0.30245945, 0.28780743, 0.40973312, 0.44427002, 0.45002616, 0.43851387, 0.61329147, 0.27315542],
[0.15705931, 0.14160401, 0.15622389, 0.29657477, 0.32456140, 0.33918129, 0.32121972, 0.42898914, 0.15747703],
[0.09419211, 0.09568131, 0.10461653, 0.20067014, 0.22114669, 0.24944155, 0.21891288, 0.31347729, 0.17684289],
[0.02568493, 0.02226027, 0.02111872, 0.02568493, 0.02397260, 0.02283105, 0.02226027, 0.05593607, 0.05079909]],

[[1.00000000, 0.00000001, 0.00000001, 0.00000001, 1.00189753, 1.00000000, 0.00000001, 0.00000001, 0.00000001],
[1.00177148, 0.45505754, 1.00000000, 1.00533333, 1.00444050, 1.00444050, 1.00000000, 0.00000001, 0.00000001],
[0.12807933, 0.09749747, 0.18128026, 0.03402707, 0.03239511, 0.04229637, 0.03309674, 0.02458675, 0.10016347],
[0.03687861, 0.03910033, 0.03517836, 0.08732055, 0.08962052, 0.07789806, 0.07770647, 0.05620428, 0.05682245],
[0.03137560, 0.03234823, 0.03137723, 0.07741438, 0.07359596, 0.07578461, 0.07256586, 0.05776795, 0.05751621],
[0.04268322, 0.04349091, 0.04298755, 0.08670228, 0.09023527, 0.11756977, 0.09316046, 0.04966192, 0.06047315],
[0.03427052, 0.03651099, 0.03591346, 0.15238085, 0.12501182, 0.14158723, 0.16502399, 0.06310204, 0.06051137],
[0.02977755, 0.03065534, 0.03267896, 0.05801752, 0.05206468, 0.05250115, 0.05332180, 0.04206739, 0.04287342],
[0.02667762, 0.04718102, 0.03795748, 0.05842002, 0.08619429, 0.12002029, 0.11735622, 0.03576535, 0.03248124]])

sep2020_school = ([[1.15122873, 0.79206049, 0.70510397, 0.16068053, 0.15122873, 0.12665406, 0.09262760, 0.06238185, 0.00189036],
[2.03975265, 3.06537102, 2.17932862, 0.21201413, 0.28886926, 0.31537102, 0.22791519, 0.19346290, 0.02738516],
[2.17346182, 2.51519644, 2.77464789, 0.47516679, 0.27724240, 0.31134173, 0.27131208, 0.30763529, 0.06375093],
[0.02823087, 0.03764115, 0.04893350, 0.06336261, 0.03826851, 0.03638645, 0.03450439, 0.09159348, 0.06336261],
[0.06181619, 0.06619256, 0.06291028, 0.04923414, 0.04759300, 0.03610503, 0.03446389, 0.04266958, 0.00382932],
[0.04500262, 0.05337520, 0.04081633, 0.02040816, 0.02668760, 0.04029304, 0.02250131, 0.01779173, 0.00104657],
[0.02255639, 0.02339181, 0.02130326, 0.00375940, 0.00417711, 0.00793651, 0.00751880, 0.01044277, 0.00543024],
[0.02010424, 0.02122115, 0.01935964, 0.01005212, 0.01042442, 0.01042442, 0.01154133, 0.01414743, 0.00148920],
[0.00342466, 0.00799087, 0.00513699, 0.01883562, 0.01084475, 0.01312785, 0.01255708, 0.03424658, 0.01712329]],

[[0.11311126, 0.09602658, 0.11583551, 0.56808688, 0.61468611, 0.54584169, 0.55286777, 0.80450642, 1.00000000],
[0.07557557, 0.07181017, 0.07653155, 0.21856199, 0.35646150, 0.23125112, 0.25695039, 0.22267402, 0.39631727],
[0.05530332, 0.05761531, 0.05922061, 0.19742364, 0.25875498, 0.25571492, 0.30818973, 0.09970169, 0.03140351],
[0.07051855, 0.07405123, 0.05054592, 0.33578347, 0.36143543, 0.35804412, 0.42575162, 0.04693660, 0.03521127],
[0.02255149, 0.02403777, 0.02268899, 0.07007864, 0.05064620, 0.05278729, 0.06735515, 0.03992342, 0.25937494],
[0.04635867, 0.06583336, 0.06313423, 0.14425982, 0.32751004, 0.21869925, 0.33051366, 0.17571392, 1.00052383],
[0.03494179, 0.04398291, 0.05565885, 0.47433102, 0.50083717, 0.40538709, 0.21454187, 0.10555745, 0.07692308],
[0.02154960, 0.02272071, 0.02078400, 0.07280244, 0.05669256, 0.05692315, 0.07402134, 0.06789706, 0.66708075],
[0.33352381, 0.16290970, 0.17652989, 0.10979871, 0.17139163, 0.11014416, 0.12371697, 0.05022488, 0.10651629]])

sep2020_community = ([[0.23818526, 0.17769376, 0.18714556, 0.11153119, 0.20982987, 0.13610586, 0.11720227, 0.11531191, 0.07750473],
[0.34893993, 0.46024735, 0.32685512, 0.08038869, 0.16784452, 0.15282686, 0.10512367, 0.12279152, 0.07597173],
[0.23573017, 0.34544107, 0.41363973, 0.16234248, 0.10674574, 0.14306894, 0.15715345, 0.07783543, 0.06819867],
[0.03952321, 0.03889586, 0.04893350, 0.36260979, 0.25031368, 0.16122961, 0.16499373, 0.17063990, 0.06336261],
[0.13074398, 0.11652079, 0.10831510, 0.28993435, 0.29540481, 0.24343545, 0.19201313, 0.27407002, 0.14277899],
[0.06331763, 0.08006279, 0.07169021, 0.28571429, 0.31240188, 0.33594976, 0.27263213, 0.29565672, 0.21245421],
[0.05973266, 0.07017544, 0.08479532, 0.24853801, 0.20426065, 0.23976608, 0.24770259, 0.24227235, 0.18170426],
[0.03946389, 0.03983619, 0.02494415, 0.22635890, 0.24534624, 0.24311243, 0.23454952, 0.32166791, 0.21407297],
[0.02340183, 0.03938356, 0.03424658, 0.18378995, 0.20433790, 0.24086758, 0.19463470, 0.28652968, 0.38299087]],

[[0.16082307, 0.13116556, 0.09515211, 0.50582925, 0.66061002, 0.57861252, 0.69740094, 0.69213908, 0.56977417],
[0.09020248, 0.13356724, 0.09309172, 0.63277201, 0.54532305, 0.64360500, 0.57578921, 0.44808932, 0.46833651],
[0.08435388, 0.09237300, 0.11100246, 0.46418295, 0.53967671, 0.59269891, 0.47639020, 0.45454837, 0.46494609],
[0.49767179, 0.47813753, 0.49910024, 0.27527364, 0.35004915, 0.31883100, 0.24619087, 0.23321263, 0.45344325],
[0.03949720, 0.03625288, 0.03257716, 0.16397993, 0.15551289, 0.14316177, 0.15497767, 0.06513346, 0.05496667],
[0.04295635, 0.05119944, 0.08096621, 0.10551320, 0.11805086, 0.11182089, 0.11946141, 0.06593968, 0.10268570],
[0.05837970, 0.05100823, 0.04339702, 0.17334033, 0.17613521, 0.19342139, 0.23427904, 0.10309586, 0.21741364],
[0.37867821, 0.38931763, 0.32939296, 0.16171187, 0.16885444, 0.16532826, 0.17121409, 0.13307671, 0.15535141],
[0.21791366, 0.48467380, 0.28837286, 0.40085742, 0.41275525, 0.58386775, 0.36779030, 0.09644326, 0.14540657]])


march2021_work = ([[0.00189036, 0.00000000, 0.00000000, 0.00378072, 0.00756144, 0.00189036, 0.00000000, 0.00000000, 0.00000000],
[0.00088339, 0.00176678, 0.00088339, 0.00353357, 0.00441696, 0.00088339, 0.00088339, 0.00176678, 0.00000000],
[0.00370645, 0.00074129, 0.00148258, 0.00370645, 0.00148258, 0.00296516, 0.00222387, 0.00074129, 0.00074129],
[0.10037641, 0.09410289, 0.09096612, 0.15244668, 0.13362610, 0.11982434, 0.10351317, 0.13927227, 0.03387704],
[0.07549234, 0.07768053, 0.08150985, 0.08205689, 0.09628009, 0.10065646, 0.08424508, 0.10831510, 0.05361050],
[0.10727368, 0.09890110, 0.10518053, 0.15489273, 0.17948718, 0.19570905, 0.18262690, 0.24385139, 0.11564626],
[0.17084378, 0.19047619, 0.18379282, 0.21846282, 0.22514620, 0.23600668, 0.24060150, 0.36758563, 0.21094403],
[0.01451973, 0.02308265, 0.01005212, 0.04206999, 0.04020849, 0.04206999, 0.04839911, 0.05994043, 0.02084885],
[0.00000000, 0.00000000, 0.00171233, 0.00114155, 0.00114155, 0.00228311, 0.00570776, 0.00171233, 0.00057078]],

[[1.00000000, 0.00000001, 0.00000001, 1.00189753, 1.00571429, 1.00000000, 0.00000001, 0.00000001, 0.00000001],
[1.00000000, 1.00088496, 1.00000000, 0.50044248, 0.71591341, 1.00000000, 1.00000000, 1.00088496, 0.00000001],
[0.29422036, 1.00000000, 1.00074239, 1.00297619, 1.00074239, 1.00223048, 1.00148588, 1.00000000, 1.00000000],
[0.02831114, 0.02726610, 0.02647316, 0.05489376, 0.04757282, 0.04564204, 0.04307730, 0.03709375, 0.10495761],
[0.03710060, 0.04822419, 0.03782676, 0.21674106, 0.16994630, 0.22944992, 0.22801447, 0.05941705, 0.07305364],
[0.06764349, 0.06196917, 0.06480773, 0.09415367, 0.08194712, 0.07968873, 0.06545321, 0.03790838, 0.03267790],
[0.03507118, 0.03561250, 0.03653829, 0.06578563, 0.07904971, 0.08516183, 0.07101641, 0.03741759, 0.03656177],
[0.16214647, 0.08106126, 0.29748007, 0.21302926, 0.20538049, 0.19948558, 0.18207948, 0.11259671, 0.12253759],
[0.00000001, 0.00000001, 1.00114351, 1.00057143, 1.00057143, 0.66730183, 0.83683808, 1.00114351, 1.00000000]])

march2021_school = ([[0.59168242, 0.40075614, 0.34215501, 0.06805293, 0.14744802, 0.11342155, 0.07183365, 0.05671078, 0.00000000],
[0.80918728, 1.45671378, 0.90724382, 0.09452297, 0.11925795, 0.14045936, 0.08745583, 0.06802120, 0.00795053],
[0.83172721, 0.98665678, 1.10007413, 0.21793921, 0.10378058, 0.13491475, 0.11267606, 0.15344700, 0.05559674],
[0.04830615, 0.05583438, 0.05144291, 0.05583438, 0.04642409, 0.03513174, 0.04328733, 0.03136763, 0.00690088],
[0.03555799, 0.04157549, 0.03501094, 0.01203501, 0.00820569, 0.00765864, 0.00765864, 0.00765864, 0.00109409],
[0.00209314, 0.00575615, 0.00052329, 0.00523286, 0.00732601, 0.00470958, 0.00418629, 0.00418629, 0.00052329],
[0.01336675, 0.01044277, 0.00668338, 0.01879699, 0.01796157, 0.01336675, 0.01754386, 0.01754386, 0.00417711],
[0.00000000, 0.00893522, 0.00000000, 0.00595681, 0.00297841, 0.00409531, 0.00335071, 0.01489203, 0.01489203],
[0.00057078, 0.00171233, 0.00114155, 0.00000000, 0.00000000, 0.00171233, 0.00114155, 0.00000000, 0.00000000]],

[[0.13619664, 0.12125755, 0.13737508, 0.58378378, 0.49118622, 0.62909567, 0.46669148, 0.57254392, 0.00000001],
[0.09234292, 0.09025222, 0.09551037, 0.53216976, 0.63353983, 0.60943489, 0.57393216, 0.48154005, 0.53117988],
[0.06022346, 0.06941702, 0.06951694, 0.24361536, 0.37258889, 0.29743465, 0.31454906, 0.06514367, 0.02512527],
[0.02770757, 0.03201698, 0.03078711, 0.08374914, 0.05970910, 0.04702801, 0.06579756, 0.09420372, 0.52538003],
[0.02560994, 0.02938612, 0.02530502, 0.73945839, 0.88830184, 0.70338797, 0.29215973, 0.35074872, 0.50000000],
[1.00157315, 1.00526316, 1.00000000, 0.62672267, 0.63900970, 0.82091691, 0.66818261, 0.36400019, 1.00000000],
[0.05352764, 0.10046500, 0.12312679, 0.07780140, 0.11014606, 0.11441276, 0.09386522, 0.09645304, 0.25015681],
[0.00000001, 0.05912252, 0.00000001, 0.26699150, 0.28585117, 0.20008943, 0.33358181, 0.02766361, 0.02500000],
[1.00000000, 0.60027425, 1.00057143, 0.00000001, 0.00000001, 1.00114351, 0.50000000, 0.00000001, 0.00000001]])

march2021_community= ([[0.13043478, 0.13799622, 0.11909263, 0.05293006, 0.08695652, 0.04914934, 0.05671078, 0.07561437, 0.04158790],
[0.17049470, 0.21113074, 0.17402827, 0.03091873, 0.06713781, 0.06095406, 0.03975265, 0.05123675, 0.01855124],
[0.25871016, 0.27798369, 0.29948110, 0.10674574, 0.09117865, 0.10378058, 0.09859155, 0.09117865, 0.06078577],
[0.01254705, 0.01317440, 0.01317440, 0.08155583, 0.05771644, 0.04328733, 0.04956085, 0.04077792, 0.02321205],
[0.01422319, 0.01531729, 0.01914661, 0.05087527, 0.06072210, 0.06400438, 0.04376368, 0.06291028, 0.02024070],
[0.04186290, 0.04971219, 0.05337520, 0.10989011, 0.10989011, 0.11930926, 0.08424908, 0.14809001, 0.11041340],
[0.06265664, 0.06390977, 0.06015038, 0.11361738, 0.11988304, 0.11904762, 0.12280702, 0.13659148, 0.09398496],
[0.00781832, 0.01005212, 0.00670141, 0.06701415, 0.07669397, 0.07967238, 0.08972450, 0.11280715, 0.07483246],
[0.00456621, 0.00913242, 0.00570776, 0.05993151, 0.06735160, 0.06506849, 0.06563927, 0.08732877, 0.11929224]],

[[0.20137300, 0.14260767, 0.14918465, 0.44702467, 0.48843663, 0.63726673, 0.70790132, 0.46983449, 0.53446213],
[0.10832543, 0.13291600, 0.10727869, 0.39774725, 0.50320813, 0.75928935, 0.49292501, 0.39956874, 0.68543476],
[0.04184204, 0.04580530, 0.04642899, 0.05145240, 0.04172362, 0.05268227, 0.04801547, 0.03807838, 0.03070016],
[0.84161031, 0.68309068, 0.57148470, 0.53819246, 0.57938265, 0.57457619, 0.55177418, 0.22068744, 0.38460936],
[0.54557357, 0.52238379, 0.25287997, 0.60251487, 0.69224625, 0.47244674, 0.51090604, 0.29949838, 0.29762120],
[0.05620362, 0.05170803, 0.04793979, 0.09686090, 0.17864158, 0.11527771, 0.16806948, 0.03587174, 0.04828387],
[0.04380800, 0.04921956, 0.04854078, 0.11603723, 0.08997074, 0.09429799, 0.09195459, 0.04649329, 0.06619062],
[0.68077271, 0.60341599, 0.50149421, 0.31350626, 0.52504528, 0.30693779, 0.27458030, 0.16035794, 0.22990237],
[1.00401376, 0.67036753, 0.71680039, 0.62202487, 0.56144365, 0.72763659, 0.67036084, 0.15901260, 0.26107872]])


polymod_work = ([[0.00000000, 0.00304414, 0.00000000, 0.00608828, 0.00608828, 0.02130898, 0.00761035, 0.00456621, 0.00152207],
[0.00000000, 0.02526316, 0.00105263, 0.00315789, 0.01368421, 0.00947368, 0.00315789, 0.00105263, 0.00105263],
[0.00813008, 0.01161440, 0.04413473, 0.08130081, 0.03484321, 0.03368177, 0.01626016, 0.00813008, 0.00116144],
[0.05116697, 0.13285458, 0.13913824, 1.06373429, 0.82675045, 0.65080790, 0.39317774, 0.14721724, 0.08617594],
[0.06756757, 0.16216216, 0.15970516, 1.05405405, 1.49631450, 1.31818182, 0.71130221, 0.20884521, 0.08722359],
[0.09745293, 0.08194906, 0.23477298, 0.88150609, 1.31893688, 1.55481728, 0.90033223, 0.25027685, 0.18936877],
[0.02444444, 0.12111111, 0.15333333, 0.58222222, 0.81444444, 1.04444444, 0.82888889, 0.19666667, 0.15000000],
[0.00699301, 0.04615385, 0.04195804, 0.19440559, 0.25594406, 0.28391608, 0.25314685, 0.09930070, 0.05174825],
[0.00000000, 0.00380228, 0.02661597, 0.04942966, 0.09505703, 0.06463878, 0.04562738, 0.03802281, 0.04182510]],

[[0.00000001, 0.50000000, 0.00000001, 0.66836475, 1.00459418, 0.31986626, 0.71709663, 0.60073260, 1.00000000],
[0.00000001, 0.07234518, 1.00000000, 1.00211193, 0.39565761, 0.82370528, 0.60050622, 1.00000000, 1.00000000],
[0.14285714, 0.15160597, 0.45012534, 0.21427148, 0.28016680, 0.43872920, 0.64227035, 0.78181818, 1.00000000],
[0.09009701, 0.10544334, 0.09016993, 0.17807637, 0.23122471, 0.27571523, 0.31307092, 0.25527951, 0.15087263],
[0.09197041, 0.09718293, 0.13151869, 0.17619503, 0.22282373, 0.23043992, 0.35322760, 0.42822883, 0.36401531],
[0.07824520, 0.27182173, 0.06397300, 0.22509802, 0.22895450, 0.21985219, 0.27763807, 0.39095097, 0.15604589],
[0.14508935, 0.05555615, 0.06524236, 0.16506834, 0.24148503, 0.19816621, 0.24547342, 0.45312113, 0.04974638],
[0.55694228, 0.06041291, 0.17144229, 0.24267662, 0.18198658, 0.25359639, 0.25062640, 0.38128977, 0.31556692],
[0.00000001, 1.00000000, 0.47073922, 0.63615988, 0.24268247, 0.47172209, 0.40588691, 0.56538628, 0.18719148]])

polymod_school = ([[1.28614916, 0.36986301, 0.04870624, 0.12328767, 0.26940639, 0.14003044, 0.06849315, 0.02130898, 0.00456621],
[0.18315789, 5.16631579, 0.44736842, 0.20526316, 0.37368421, 0.36842105, 0.20842105, 0.03789474, 0.00421053],
[0.03019744, 0.24390244, 6.76190476, 0.59814170, 0.43089431, 0.45644599, 0.23577236, 0.03252033, 0.00348432],
[0.02962298, 0.08886894, 0.30251346, 1.52154399, 0.15080790, 0.14003591, 0.07450628, 0.01974865, 0.00359066],
[0.06879607, 0.18427518, 0.02579853, 0.11179361, 0.13882064, 0.09705160, 0.02334152, 0.00982801, 0.00000000],
[0.09080842, 0.11295681, 0.26688815, 0.11074197, 0.09966777, 0.11960133, 0.05980066, 0.01439646, 0.00110742],
[0.05666667, 0.29888889, 0.20000000, 0.07555556, 0.05000000, 0.07888889, 0.06111111, 0.01000000, 0.00333333],
[0.02937063, 0.03356643, 0.03356643, 0.03216783, 0.04055944, 0.02377622, 0.02237762, 0.03776224, 0.01118881],
[0.00000000, 0.00760456, 0.00760456, 0.00760456, 0.00000000, 0.00000000, 0.01140684, 0.00000000, 0.00000000]],

[[0.08001684, 0.08916543, 0.13288599, 0.57353798, 0.27471857, 0.46133619, 0.49969531, 0.54388251, 1.00305810],
[0.14728612, 0.09521596, 0.12662371, 0.68355342, 0.52432979, 0.70625133, 0.66656497, 0.73941650, 1.00317125],
[0.13303189, 0.18484288, 0.09340819, 0.18391482, 0.53617222, 0.49214559, 0.59437560, 0.79703429, 1.00233100],
[0.06105372, 0.07261121, 0.09386465, 0.08310334, 0.48983569, 0.48380517, 0.56482608, 0.85976124, 0.66766647],
[0.24092968, 0.11592426, 0.45146363, 0.19303958, 0.28406094, 0.34107963, 0.92314588, 0.80534918, 0.00000001],
[0.08105377, 0.08348123, 0.04121662, 0.09089718, 0.17800992, 0.32782119, 0.31960630, 0.77236201, 1.00000000],
[0.09609955, 0.03853672, 0.03984929, 0.11229738, 0.25722461, 0.26522590, 0.34414477, 0.60295104, 0.60053440],
[0.10063492, 0.29517346, 0.07852842, 0.08498416, 0.24168359, 0.82425642, 0.44828127, 0.18726323, 0.25035063],
[0.00000001, 1.00383142, 1.00383142, 1.00383142, 0.00000001, 0.00000001, 0.60183767, 0.00000001, 0.00000001]])

polymod_community = ([[0.59056317, 0.46118721, 0.15220700, 0.54337900, 0.79299848, 0.45814307, 0.42922374, 0.32420091, 0.17503805],
[0.23263158, 1.90421053, 0.55263158, 0.40526316, 0.67894737, 0.59473684, 0.26315789, 0.22210526, 0.15263158],
[0.06039489, 0.45528455, 2.96399535, 1.09407666, 0.50058072, 0.64808362, 0.22415796, 0.13472706, 0.11614402],
[0.10323160, 0.16247756, 0.44254937, 2.87163375, 0.96409336, 0.69030521, 0.42818671, 0.14272890, 0.12387792],
[0.19901720, 0.31941032, 0.13759214, 0.92260442, 1.54545455, 0.99017199, 0.57371007, 0.36117936, 0.19041769],
[0.07308970, 0.18936877, 0.29125138, 0.75193798, 1.00110742, 1.16722038, 0.59468439, 0.28792913, 0.26135105],
[0.07111111, 0.15777778, 0.10888889, 0.86555556, 0.86333333, 0.91666667, 1.03666667, 0.57222222, 0.32666667],
[0.08111888, 0.11888112, 0.06853147, 0.59300699, 0.90629371, 0.84195804, 0.84615385, 0.94965035, 0.57342657],
[0.04562738, 0.08365019, 0.11406844, 0.36501901, 0.47148289, 0.64258555, 0.55513308, 0.92395437, 0.92395437]],

[[0.25077293, 0.38724459, 0.57127928, 0.39561732, 0.42093129, 0.58120765, 0.62716381, 0.52390272, 0.58325087],
[0.42818527, 0.17402220, 0.22461538, 0.46394083, 0.48846284, 0.42726458, 0.62714777, 0.61968749, 0.54276699],
[0.80080223, 0.16155978, 0.16752468, 0.22299626, 0.47036280, 0.48198434, 0.62642945, 0.59610168, 0.53020962],
[0.42692181, 0.17196555, 0.21223394, 0.17900824, 0.32550134, 0.54024318, 0.55495671, 0.73440956, 0.46752749],
[0.55457026, 0.36525688, 0.56901465, 0.38227827, 0.32653688, 0.42415341, 0.53917290, 0.54195576, 0.57446401],
[0.25050495, 0.38245358, 0.28296908, 0.37175805, 0.40657332, 0.39229917, 0.42516023, 0.49082066, 0.34448540],
[0.58409811, 0.30334382, 0.52820212, 0.34726793, 0.47786076, 0.40620251, 0.43432178, 0.43290867, 0.37652358],
[0.67904109, 0.52107839, 0.79380133, 0.47974594, 0.43004212, 0.49472276, 0.44849246, 0.37918305, 0.37948691],
[0.68496732, 0.51984127, 0.54758256, 0.46660730, 0.53816592, 0.46455850, 0.64086584, 0.24790705, 0.23583144]])


am2021_work = ([[0.00000000, 0.00000000, 0.00000000, 0.00567108, 0.01890359, 0.00000000, 0.00000000, 0.00000000, 0.00189036],
[0.00000000, 0.00088339, 0.00088339, 0.00441696, 0.00441696, 0.00265018, 0.00265018, 0.00176678, 0.00000000],
[0.01779096, 0.01482580, 0.01556709, 0.02149741, 0.01111935, 0.01482580, 0.00889548, 0.01408451, 0.01037806],
[0.12484316, 0.14554580, 0.11794228, 0.32496863, 0.29548306, 0.26976161, 0.26035132, 0.43161857, 0.19322459],
[0.20185996, 0.19967177, 0.20842451, 0.28829322, 0.29595186, 0.29376368, 0.27461707, 0.32494530, 0.12855580],
[0.10465725, 0.13605442, 0.12401884, 0.21978022, 0.22762951, 0.28728414, 0.23390895, 0.27420199, 0.13396128],
[0.20802005, 0.23182957, 0.24394319, 0.25271512, 0.25020886, 0.26858814, 0.27527151, 0.37928154, 0.16583124],
[0.10573343, 0.09605361, 0.09865972, 0.10945644, 0.10647803, 0.11206255, 0.12248697, 0.17870439, 0.07706627],
[0.00570776, 0.00627854, 0.00742009, 0.01027397, 0.00627854, 0.01084475, 0.00856164, 0.01084475, 0.00285388]],

[[0.00000001, 0.00000001, 0.00000001, 1.00380228, 0.72269368, 0.00000001, 0.00000001, 0.00000001, 1.00000000],
[0.00000001, 1.00000000, 1.00000000, 0.71591341, 1.00354925, 1.00177148, 1.00177148, 1.00088496, 0.00000001],
[0.08281921, 0.15408356, 0.13060328, 0.11215608, 0.24639306, 0.25719792, 0.37597620, 0.09183813, 0.07142857],
[0.03548708, 0.03590060, 0.03505376, 0.08368945, 0.06338748, 0.07195748, 0.07572451, 0.03911740, 0.04824189],
[0.03214624, 0.03451241, 0.03412236, 0.06951391, 0.07381736, 0.07137764, 0.07296611, 0.04512243, 0.04124322],
[0.04361605, 0.04659878, 0.05074891, 0.07564027, 0.07989230, 0.07486583, 0.07311518, 0.04868101, 0.04722775],
[0.03938724, 0.04043401, 0.03918474, 0.09845925, 0.10020708, 0.11355081, 0.11444944, 0.05363678, 0.04392108],
[0.03896450, 0.04444154, 0.03861054, 0.06510906, 0.07690380, 0.08245156, 0.08666755, 0.04328087, 0.05033339],
[0.10000000, 0.09090909, 0.11930356, 0.37623550, 0.47942750, 0.57904447, 0.71827057, 0.40580364, 1.00228964]])

am2021_school = ([[0.76181474, 0.40075614, 0.32892250, 0.09829868, 0.14933837, 0.09073724, 0.06805293, 0.04536862, 0.00567108],
[1.53091873, 2.56360424, 1.77208481, 0.16696113, 0.25176678, 0.23409894, 0.15017668, 0.13692580, 0.01590106],
[1.73832468, 1.97405486, 2.21497405, 0.52483321, 0.27650111, 0.31430689, 0.27575982, 0.26686434, 0.03780578],
[0.11041405, 0.13488080, 0.13237139, 0.11417817, 0.07465496, 0.06273526, 0.05897114, 0.12860728, 0.08594730],
[0.05470460, 0.05743982, 0.05579869, 0.03282276, 0.02789934, 0.03118162, 0.02352298, 0.02899344, 0.00820569],
[0.01622187, 0.02040816, 0.01936159, 0.01412873, 0.01308216, 0.01988488, 0.01360544, 0.01674516, 0.00627943],
[0.00375940, 0.03341688, 0.00960735, 0.01086048, 0.00626566, 0.00751880, 0.00751880, 0.00334169, 0.00083542],
[0.02010424, 0.02122115, 0.02010424, 0.00670141, 0.00409531, 0.00521221, 0.00521221, 0.01005212, 0.00409531],
[0.00000000, 0.00228311, 0.00057078, 0.00342466, 0.00399543, 0.00570776, 0.00285388, 0.01655251, 0.01255708]],

[[0.11659116, 0.13537676, 0.15980518, 0.52485089, 0.55062439, 0.68791054, 0.78884462, 0.77495108, 1.00380228],
[0.12067738, 0.10251726, 0.11266545, 0.61420235, 0.65123050, 0.47881237, 0.56154278, 0.49991873, 0.29141139],
[0.07045575, 0.07206747, 0.08051919, 0.18833771, 0.14874045, 0.16874974, 0.14932178, 0.11089427, 0.07594148],
[0.03568777, 0.03996747, 0.04447680, 0.21367674, 0.20500268, 0.17298370, 0.23458473, 0.03989989, 0.03581114],
[0.02762042, 0.03214198, 0.02853735, 0.05984565, 0.09810305, 0.07727492, 0.08699894, 0.08181314, 0.15801309],
[0.23036759, 0.20280314, 0.17420392, 0.12812677, 0.32588963, 0.26514211, 0.22088586, 0.20840863, 0.33385772],
[0.33361216, 0.04003681, 0.22368758, 0.22459102, 0.45565330, 0.33403127, 0.33403127, 0.80180935, 1.00041806],
[0.03937510, 0.03686061, 0.03960623, 0.39218709, 0.58010724, 0.78065293, 0.30471790, 0.24374868, 0.37975904],
[0.00000001, 0.66730183, 1.00000000, 0.60089224, 0.46727155, 0.41742157, 0.71533622, 0.16997098, 0.15514297]])

am2021_community= ([[0.13043478, 0.13610586, 0.07183365, 0.09640832, 0.13232514, 0.10775047, 0.09640832, 0.11153119, 0.04158790],
[0.26590106, 0.35954064, 0.28268551, 0.06095406, 0.09098940, 0.09010601, 0.06537102, 0.06537102, 0.04593640],
[0.42994811, 0.49369904, 0.47368421, 0.17049666, 0.12527798, 0.14010378, 0.13046701, 0.13639733, 0.05189029],
[0.06022585, 0.06712673, 0.07716437, 0.23400251, 0.18005019, 0.14868256, 0.13550816, 0.21329987, 0.13927227],
[0.01859956, 0.02571116, 0.02571116, 0.11870897, 0.14606127, 0.11487965, 0.09628009, 0.09354486, 0.04376368],
[0.03506018, 0.05651491, 0.03506018, 0.13710099, 0.14965986, 0.16640502, 0.15175301, 0.14495029, 0.11145997],
[0.04678363, 0.04344194, 0.03842941, 0.14703425, 0.14076859, 0.14578112, 0.16499582, 0.15497076, 0.14160401],
[0.04579300, 0.04393150, 0.03760238, 0.14147431, 0.15003723, 0.15897245, 0.14966493, 0.23454952, 0.15562174],
[0.00742009, 0.01369863, 0.01655251, 0.08504566, 0.09874429, 0.10616438, 0.09988584, 0.12328767, 0.14897260]],

[[0.24596273, 0.17184082, 0.23518380, 0.54104882, 0.53441296, 0.61267864, 0.60539568, 0.41937481, 0.70639747],
[0.14059803, 0.17284234, 0.15127096, 0.74292433, 0.84609574, 0.80364756, 0.76093756, 0.70305097, 0.59300637],
[0.08799189, 0.09634970, 0.09865500, 0.09175615, 0.09188821, 0.21546549, 0.10613489, 0.06303384, 0.14035443],
[0.11205060, 0.09532110, 0.11802381, 0.18365417, 0.18183941, 0.13323003, 0.14859228, 0.04258257, 0.04586665],
[0.53625816, 0.29033829, 0.11633301, 0.54323517, 0.61486850, 0.53559164, 0.41776928, 0.20321721, 0.35935564],
[0.09215558, 0.09655947, 0.10075204, 0.15366824, 0.11642663, 0.15098386, 0.12450571, 0.05211047, 0.06390633],
[0.09292572, 0.07772557, 0.11100243, 0.15177564, 0.19499122, 0.20257226, 0.27808677, 0.09568945, 0.09847683],
[0.06446278, 0.08267083, 0.05984013, 0.09392582, 0.18207620, 0.18201254, 0.13715802, 0.06964711, 0.07541024],
[0.28934423, 0.17173401, 0.15539058, 0.53121647, 0.58428923, 0.72542966, 0.73635959, 0.28503473, 0.41136029]])

contact_scenarios = {"September2020":  {"w": (age_bins,sep2020_work),
                                   "s": (age_bins,sep2020_school),
                                   "c": (age_bins,sep2020_community)},
                     "March2021": {"w": (age_bins,march2021_work),
                                   "s": (age_bins,march2021_school),
                                   "c": (age_bins,march2021_community)},
                     "AprilMay2021": {"w": (age_bins,am2021_work),
                                      "s": (age_bins,am2021_school),
                                      "c": (age_bins,am2021_community)},
                     "Polymod":   {"w": (age_bins,polymod_work),
                                   "s": (age_bins,polymod_school),
                                   "c": (age_bins,polymod_community)}}
