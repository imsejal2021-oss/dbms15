import os
import shutil
import time
import random
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='pySimpleDB Benchmark — Query Optimization & Indexing',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--query', type=str, default='all',
                        choices=['Q1', 'Q2', 'Q3', 'all'],
                        help='Which query to run (default: all)')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'opt', 'index', 'full'],
                        help=(
                            'Execution mode:\n'
                            '  baseline — No optimization, no indexes (default)\n'
                            '  opt      — Join reordering + selection pushdown, no indexes\n'
                            '  index    — Indexes only, original join order\n'
                            '  full     — Both optimization and indexes'
                        ))
    args = parser.parse_args()

    # =========================================================================
    # Database Setup
    # =========================================================================
    db_dir = os.path.join(os.getcwd(), 'benchmarkdb')
    if os.path.exists(db_dir):
        print(f"Removing existing database directory: {db_dir}")
        shutil.rmtree(db_dir)

    from FileSystem import FileMgr
    from BufferPool import LogMgr, BufferMgr
    from Transaction import Transaction
    from Metadata import MetadataMgr
    from Planner import BasicQueryPlanner, BasicUpdatePlanner, Planner
    from Record import Schema, TableScan

    class BenchmarkDB:
        def __init__(self, db_name, block_size, buffer_pool_size):
            self.fm = FileMgr(db_name, block_size)
            self.lm = LogMgr(self.fm, db_name + '.log')
            self.bm = BufferMgr(self.fm, self.lm, buffer_pool_size)
    
            tx = Transaction(self.fm, self.lm, self.bm)
            if self.fm.db_exists:
                print('Recovering...')
                tx.recover()
                self.mm = MetadataMgr(tx, False)
            else:
                print('Created new db...')
                self.mm = MetadataMgr(tx, True)
            tx.commit()

    print("Initializing Database...")
    db = BenchmarkDB('benchmarkdb', 8192, 1000)
    tx = Transaction(db.fm, db.lm, db.bm)

    # =========================================================================
    # Schema Creation
    # =========================================================================
    print("Creating schemas...")
    sStudent = Schema(['s_id', 'int', 4], ['s_name', 'str', 50], ['s_department', 'str', 30], ['s_year', 'int', 4])
    db.mm.createTable(tx, 'Student', sStudent)

    sInstructor = Schema(['i_id', 'int', 4], ['i_name', 'str', 50], ['i_department', 'str', 30])
    db.mm.createTable(tx, 'Instructor', sInstructor)

    sCourse = Schema(['c_id', 'int', 4], ['c_title', 'str', 100], ['c_department', 'str', 30], ['c_credits', 'int', 4])
    db.mm.createTable(tx, 'Course', sCourse)

    sSection = Schema(['sec_id', 'int', 4], ['sec_course_id', 'int', 4], ['sec_instructor_id', 'int', 4], ['sec_semester', 'str', 10], ['sec_year', 'int', 4])
    db.mm.createTable(tx, 'Section', sSection)

    sEnrollment = Schema(['e_id', 'int', 4], ['e_student_id', 'int', 4], ['e_section_id', 'int', 4], ['e_grade', 'str', 2])
    db.mm.createTable(tx, 'Enrollment', sEnrollment)

    # =========================================================================
    # Data Population
    # =========================================================================
    random.seed(42)

    departments = ['CS', 'EE', 'ME', 'Math', 'Physics', 'Biology', 'Chemistry', 'English', 'History']
    semesters = ['Fall', 'Spring', 'Summer']
    grades = ['A', 'B', 'C', 'D', 'F', 'NC']

    print("Populating database...")
    
    print("Inserting Students (100)...")
    ts_student = TableScan(tx, 'Student', db.mm.getLayout(tx, 'Student'))
    for i in range(1, 101):
        ts_student.nextEmptyRecord()
        ts_student.setInt('s_id', i)
        ts_student.setString('s_name', f"Student_{i}")
        ts_student.setString('s_department', random.choice(departments))
        ts_student.setInt('s_year', random.randint(2021, 2025))
    ts_student.closeRecordPage()

    print("Inserting Instructors (50)...")
    ts_instructor = TableScan(tx, 'Instructor', db.mm.getLayout(tx, 'Instructor'))
    for i in range(1, 51):
        ts_instructor.nextEmptyRecord()
        ts_instructor.setInt('i_id', i)
        ts_instructor.setString('i_name', f"Instructor_{i}")
        ts_instructor.setString('i_department', random.choice(departments))
    ts_instructor.closeRecordPage()

    print("Inserting Courses (20)...")
    ts_course = TableScan(tx, 'Course', db.mm.getLayout(tx, 'Course'))
    for i in range(1, 21):
        ts_course.nextEmptyRecord()
        ts_course.setInt('c_id', i)
        ts_course.setString('c_title', f"Course_{i}")
        dept = random.choice([d for d in departments])
        ts_course.setString('c_department', dept)
        ts_course.setInt('c_credits', random.choice([3, 4]))
    ts_course.closeRecordPage()

    print("Inserting Sections (300)...")
    ts_section = TableScan(tx, 'Section', db.mm.getLayout(tx, 'Section'))
    for i in range(1, 301):
        ts_section.nextEmptyRecord()
        ts_section.setInt('sec_id', i)
        course_id = random.randint(1, 20)
        ts_section.setInt('sec_course_id', course_id)
        ts_section.setInt('sec_instructor_id', random.randint(1, 50))
        ts_section.setString('sec_semester', random.choice(semesters))
        ts_section.setInt('sec_year', random.randint(2021,2025))
    ts_section.closeRecordPage()

    print("Inserting Enrollments (500)...")
    ts_enrollment = TableScan(tx, 'Enrollment', db.mm.getLayout(tx, 'Enrollment'))
    for i in range(1, 501):
        ts_enrollment.nextEmptyRecord()
        ts_enrollment.setInt('e_id', i)
        ts_enrollment.setInt('e_student_id', random.randint(1, 100))
        ts_enrollment.setInt('e_section_id', random.randint(1, 300))
        ts_enrollment.setString('e_grade', random.choice(grades))
    ts_enrollment.closeRecordPage()

    tx.commit()
    print("Database populated successfully!\n")

    # =========================================================================
    # Index Definitions
    # =========================================================================
    index_defs = {
        'Student':    [('s_id',              'int', 4)],
        'Enrollment': [('e_student_id',      'int', 4),
                       ('e_section_id',      'int', 4),
                       ('e_grade',           'str', 2)],
        'Section':    [('sec_id',            'int', 4),
                       ('sec_course_id',     'int', 4),
                       ('sec_instructor_id', 'int', 4),
                       ('sec_semester',      'str', 10),
                       ('sec_year',          'int', 4)],
        'Course':     [('c_id',              'int', 4),
                       ('c_department',      'str', 30)],
        'Instructor': [('i_id',              'int', 4)],
    }

    composite_index_defs = {
        'Section': [( ('sec_semester', 'sec_year'), ('str', 'int'), (10, 4) )]
    }

    # =========================================================================
    # Index Creation (for 'index' and 'full' modes)
    # =========================================================================
    indexes = None
    if args.mode in ('index', 'full'):
        print("Building indexes...")
        from solution import create_indexes
        tx_idx = Transaction(db.fm, db.lm, db.bm)
        indexes = create_indexes(db, tx_idx, index_defs, composite_index_defs)
        tx_idx.commit()
        print("Indexes built successfully!\n")

    # =========================================================================
    # Planner Setup
    # =========================================================================
    tx2 = Transaction(db.fm, db.lm, db.bm)
    up = BasicUpdatePlanner(db.mm)

    if args.mode == 'baseline':
        qp = BasicQueryPlanner(db.mm)
    elif args.mode == 'opt':
        from solution import BetterQueryPlanner
        qp = BetterQueryPlanner(db.mm)
    elif args.mode == 'index':
        from solution import IndexQueryPlanner
        qp = IndexQueryPlanner(db.mm, indexes)
    elif args.mode == 'full':
        from solution import BetterQueryPlanner, IndexQueryPlanner
        # Full mode: optimized planner wrapped with index support
        # IndexQueryPlanner uses both optimization and indexes
        better = BetterQueryPlanner(db.mm)
        qp = IndexQueryPlanner(db.mm, indexes, better_planner = better)

    p = Planner(qp, up)

    # =========================================================================
    # Query Definitions
    # =========================================================================
    all_queries = [
        ("Q1", "select s_id, s_name from Student, Enrollment, Section, Course where s_id = e_student_id and e_section_id = sec_id and sec_course_id = c_id and c_department = 'CS'"),
        ("Q2", "select s_id, s_name from Student, Enrollment where s_id = e_student_id and e_grade = 'NC'"),
        ("Q3", "select i_id, i_name from Instructor, Section where i_id = sec_instructor_id and sec_semester = 'Fall' and sec_year = 2024"),
    ]

    if args.query == 'all':
        queries = all_queries
    else:
        queries = [q for q in all_queries if q[0] == args.query]

    # =========================================================================
    # Query Execution
    # =========================================================================
    mode_descriptions = {
        'baseline': 'Baseline (no optimization, no indexes)',
        'opt':      'Optimized (join reordering + selection pushdown)',
        'index':    'Indexed (B-tree indexes, original join order)',
        'full':     'Full (optimization + indexes)',
    }

    print("=" * 70)
    print(f"  MODE: {mode_descriptions[args.mode]}")
    print(f"  QUERIES: {args.query}")
    print("=" * 70)

    results = []

    for name, q in queries:
        print(f"\n{'-' * 60}")
        print(f"  {name} | Mode: {args.mode}")
        print(f"  Query: {q}")
        print(f"{'-' * 60}")

        # Reset I/O counters before each query
        db.fm.reset_counters()

        start_time = time.time()
        
        try:
            plan = p.createQueryPlan(tx2, q)
            scan = plan.open()
            count = 0
            while scan.nextRecord():
                count += 1
            scan.closeRecordPage()
            
            end_time = time.time()
            elapsed = end_time - start_time
            block_accesses = db.fm.block_accesses

            print(f"  [OK] Rows returned   : {count}")
            print(f"  [OK] Time            : {elapsed:.4f} seconds")
            print(f"  [OK] Block accesses  : {block_accesses}")


            results.append({
                'query': name, 'mode': args.mode,
                'rows': count, 'time': elapsed,
                'accesses': block_accesses,
            })

        except NotImplementedError as e:
            end_time = time.time()
            print(f"  [!!] NOT IMPLEMENTED: {e}")
            print(f"  (Time before error: {end_time - start_time:.4f}s)")
        except Exception as e:
            end_time = time.time()
            print(f"  [!!] FAILED: {e}")
            print(f"  (Time before error: {end_time - start_time:.4f}s)")
            
    tx2.commit()

    # =========================================================================
    # Summary Table
    # =========================================================================
    if results:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY — Mode: {args.mode}")
        print(f"{'=' * 70}")
        print(f"  {'Query':<8} {'Rows':<8} {'Time (s)':<12} {'Accesses':<12}")
        print(f"  {'-' * 60}")
        for r in results:
            print(f"  {r['query']:<8} {r['rows']:<8} {r['time']:<12.4f} {r['accesses']:<12}")
        print()


if __name__ == "__main__":
    main()
