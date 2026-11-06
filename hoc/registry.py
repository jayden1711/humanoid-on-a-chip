"""Policy spec strings -> policy objects (so worker processes can build their own copy).

teacher            original float32 LSTM
ptqW<b>            teacher with weights rounded to b bits (float arithmetic)
float:<file.npz>   distilled float student (hoc.student)
int:<file.bin>     packed integer model run by the integer reference (hoc.intref)
"""
from .policies import TeacherLSTM, TeacherFakeQuant


def load_policy(spec):
    if spec == "teacher":
        return TeacherLSTM()
    if spec.startswith("ptqW"):
        return TeacherFakeQuant(int(spec[4:]))
    if spec.startswith("float:"):
        from .student import FloatStudent
        return FloatStudent.load(spec[6:])
    if spec.startswith("int:"):
        from .intref import IntPolicy
        return IntPolicy.from_file(spec[4:])
    raise KeyError(spec)
