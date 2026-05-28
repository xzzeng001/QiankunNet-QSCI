数据结果共有两个npy格式文件

single_qubit.npy：
	将单比特信息存为字典，key为比特index，0~72；
	value为信息字典：
		qubit: str, 对应芯片上物理比特编号；
		xeb_1q: float，单比特XEB；
		spb_1q: float，单比特SPB；
		readout_f00: float，单比特读取F00；
		readout_f11: float，单比特读取F11；
		event_state: list[bool]，共计一亿次测量结果，True为1态，False为0态；

cz_xeb.npy:
	将CZ信息存为字典，key为比特index组合，(q0,q1):
	value为信息字典：
		coupler: str，对应芯片上coupler编号；
		q0: int，比特0的index；
		q1: int，比特1的index；
		xeb: float，CZ XEB；
		spb: float，CZ SPB；

QCIS_circuit_qubit_mapping.txt:
	QASM Qubit 与物理比特映射关系:
	QASM Qubit: 实验上的物理比特

QCIS_circuit_coupler_mapping.txt:
	QASM coupler 与物理coupler映射关系:
	QASM coupler: 实验上的物理比特

量子线路：
	单比特门数目：693 (不包含RZ)；
	CZ门数目：146；
	CZ层数：6；
