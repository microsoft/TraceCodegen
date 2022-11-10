from execution.program_tracing import get_execution_states
from lightning_modules.models.gpt_stmt_partial_mml_model import construct_program_from_trace

def main():
    sample_program = "n0=1\nn1=2\nn2=3\nn3=4\nt0=n0+n1\nt1=t0+n2\nt2=n0+t1\nans=t2+n3"

    trace = get_execution_states(sample_program)

    program = construct_program_from_trace(trace)

    print(program)

if __name__ == "__main__":
    main()